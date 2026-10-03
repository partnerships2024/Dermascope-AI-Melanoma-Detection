"""
Training pipeline for Dermascope AI with two-phase training and mixed precision.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
import pandas as pd

from .model import DermaScope
from .dataset import SkinLesionDataset, get_transforms, get_patient_split, get_weighted_sampler
from .utils import set_seed, setup_logger, EarlyStopping
from .config import DATA_DIR, MODEL_DIR, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, METADATA_DIM

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced datasets.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha # Can be class weights
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none', label_smoothing=0.1)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc='Training')
    for images, metadata, targets in pbar:
        images, metadata, targets = images.to(device), metadata.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision
        with autocast():
            outputs = model(images, metadata)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
        
    epoch_loss = running_loss / len(dataloader)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    return epoch_loss, bal_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for images, metadata, targets in tqdm(dataloader, desc='Validation'):
            images, metadata, targets = images.to(device), metadata.to(device), targets.to(device)
            
            outputs = model(images, metadata)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    
    # Calculate ROC-AUC if possible
    try:
        roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
    except ValueError:
        roc_auc = 0.0
        
    return epoch_loss, bal_acc, roc_auc

def main(args):
    set_seed(args.seed)
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger = setup_logger('train', os.path.join(MODEL_DIR, 'train.log'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    df = pd.read_csv(os.path.join(DATA_DIR, 'HAM10000_metadata.csv'))
    train_df, val_df = get_patient_split(df)
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    image_dir = os.path.join(DATA_DIR, 'HAM10000_images')
    train_dataset = SkinLesionDataset(train_df, image_dir, transforms=get_transforms('train'), apply_hair_removal=True)
    val_dataset = SkinLesionDataset(val_df, image_dir, transforms=get_transforms('val'), apply_hair_removal=False)
    
    # Handling class imbalance
    sampler = get_weighted_sampler(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model
    model = DermaScope(num_classes=NUM_CLASSES, metadata_dim=METADATA_DIM).to(device)
    
    # Calculate class weights for Focal Loss
    class_counts = train_df['dx'].value_counts().sort_index().values
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    weights = weights / weights.sum() * NUM_CLASSES
    criterion = FocalLoss(alpha=weights.to(device), gamma=2.0)
    
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=7, path=os.path.join(MODEL_DIR, 'best_model.pt'))
    
    # Phase 1: Train Head Only
    logger.info("Starting Phase 1: Training Head Only")
    model.freeze_backbone()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    
    for epoch in range(10):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, val_auc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        logger.info(f"Epoch {epoch+1}/10 - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        early_stopping(val_loss, model)
        
    # Phase 2: Fine-tuning
    logger.info("Starting Phase 2: Fine-tuning last 3 layers")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt')))
    model.unfreeze_backbone(num_layers=3)
    
    # Different learning rates for backbone and head
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.film.parameters(), 'lr': 5e-4},
        {'params': model.head.parameters(), 'lr': 5e-4}
    ]
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    early_stopping = EarlyStopping(patience=7, path=os.path.join(MODEL_DIR, 'best_model_finetuned.pt'))
    
    for epoch in range(25):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, val_auc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        logger.info(f"Epoch {epoch+1}/25 - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered in Phase 2")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DermaScope Model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    main(args)
