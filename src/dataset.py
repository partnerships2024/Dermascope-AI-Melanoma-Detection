"""
Dataset implementation for the HAM10000 skin lesion dataset.
"""
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupShuffleSplit
from .preprocessing import dull_razor

from .config import CLASS_NAMES, CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for HAM10000 with image and metadata loading.
    """
    def __init__(self, df: pd.DataFrame, image_dir: str, transforms: Optional[A.Compose] = None, apply_hair_removal: bool = False):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.apply_hair_removal = apply_hair_removal
        
        # Preprocess metadata
        self.metadata = self._preprocess_metadata(df)
        
    def _preprocess_metadata(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocesses metadata: age imputation and one-hot encoding.
        """
        # Age imputation with median
        median_age = df['age'].median()
        age = df['age'].fillna(median_age).values / 100.0  # Scale age
        
        # One-hot encoding for sex and localization
        # Assuming typical unique values from HAM10000
        sex_dummies = pd.get_dummies(df['sex'], dummy_na=True)
        loc_dummies = pd.get_dummies(df['localization'], dummy_na=True)
        
        # Combine metadata
        metadata = np.column_stack((age, sex_dummies.values, loc_dummies.values))
        return metadata.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_id = row['image_id']
        label_str = row['dx']
        label = CLASS_TO_IDX.get(label_str, 0)
        
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if self.apply_hair_removal:
            image = dull_razor(image)
            
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
            
        meta = torch.tensor(self.metadata[idx], dtype=torch.float32)
        
        return image, meta, label

def get_transforms(phase: str) -> A.Compose:
    """Returns Albumentations transforms for train/val phases."""
    if phase == 'train':
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])

def get_patient_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Patient-level stratified split to prevent data leakage.
    Uses 'lesion_id' to group images from the same patient.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(df, df['dx'], groups=df['lesion_id']))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

def get_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler to handle class imbalance.
    """
    class_counts = df['dx'].value_counts()
    class_weights = 1.0 / class_counts
    sample_weights = df['dx'].map(class_weights).values
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
