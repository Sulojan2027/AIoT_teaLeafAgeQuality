import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data_utils.dataset import TeaLeafDataset

# Standard ImageNet statistics for normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_transforms(split):
    if split == 'train':
        return A.Compose([
            A.RandomResizedCrop((300, 300), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                p=0.5
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(300, 300),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])

def create_dataloaders(data_dir='data', batch_size=32, num_workers=2):
    """
    Creates and returns train, validation, and test dataloaders.
    """
    print(f"Initializing Datasets from {data_dir}...")

    train_transform = get_transforms(split="train")
    val_transform = get_transforms(split="valid")
    
    train_dataset = TeaLeafDataset(data_dir=data_dir, split='train', transform=train_transform)
    valid_dataset = TeaLeafDataset(data_dir=data_dir, split='valid', transform=val_transform)
    test_dataset = TeaLeafDataset(data_dir=data_dir, split='test', transform=val_transform)

    # Create DataLoaders
    # shuffle=True only for training!
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Train Loader: {len(train_loader)} batches")
    print(f"Valid Loader: {len(valid_loader)} batches")
    print(f"Test Loader:  {len(test_loader)} batches")

    return train_loader, valid_loader, test_loader
