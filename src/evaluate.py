"""
Evaluation script to generate metrics, confusion matrix, and ROC curves.
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import DermaScope
from .dataset import SkinLesionDataset, get_transforms, get_patient_split
from .config import DATA_DIR, MODEL_DIR, RESULTS_DIR, CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES, METADATA_DIM

def evaluate_model():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = DermaScope(num_classes=NUM_CLASSES, metadata_dim=METADATA_DIM).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model_finetuned.pt'), map_location=device))
    model.eval()
    
    # Load validation data
    df = pd.read_csv(os.path.join(DATA_DIR, 'HAM10000_metadata.csv'))
    _, val_df = get_patient_split(df)
    
    image_dir = os.path.join(DATA_DIR, 'HAM10000_images')
    val_dataset = SkinLesionDataset(val_df, image_dir, transforms=get_transforms('val'), apply_hair_removal=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, metadata, targets in tqdm(val_loader):
            images, metadata = images.to(device), metadata.to(device)
            outputs = model(images, metadata)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 1. Classification Report
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES)
    print("\nClassification Report:\n", report)
    with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    # Melanoma-specific metrics (mel is class index corresponding to 'mel')
    mel_idx = CLASS_TO_IDX['mel']
    mel_true = (all_targets == mel_idx)
    mel_pred = (all_preds == mel_idx)
    
    tn, fp, fn, tp = confusion_matrix(mel_true, mel_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn+fp) > 0 else 0
    
    metrics_str = f"Melanoma Sensitivity: {sensitivity:.4f}\nMelanoma Specificity: {specificity:.4f}\n"
    metrics_str += f"Balanced Accuracy: {balanced_accuracy_score(all_targets, all_preds):.4f}\n"
    metrics_str += f"Weighted ROC-AUC: {roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted'):.4f}\n"
    
    print(metrics_str)
    with open(os.path.join(RESULTS_DIR, 'metrics.txt'), 'w') as f:
        f.write(metrics_str)
        
    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # 3. ROC Curves per class
    plt.figure(figsize=(10, 8))
    for i, cls_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve((all_targets == i).astype(int), all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{cls_name} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic per Class')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves.png'))
    plt.close()

if __name__ == '__main__':
    evaluate_model()
