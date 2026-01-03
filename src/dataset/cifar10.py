"""
CIFAR-10 dataset wrapper for diffusion models.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from typing import Optional, Tuple
from pathlib import Path

from .transforms import get_transform, get_inference_transform


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 dataset wrapper for diffusion model training.
    
    Returns images normalized to [-1, 1] range.
    """
    
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        image_size: int = 32,
        download: bool = True,
        augment: bool = True,
    ):
        """
        Initialize CIFAR-10 dataset.
        
        Args:
            root: Root directory for dataset
            train: Load training set (True) or test set (False)
            image_size: Target image size (CIFAR-10 is natively 32x32)
            download: Download dataset if not found
            augment: Apply data augmentation (random flip)
        """
        self.root = Path(root)
        self.train = train
        self.image_size = image_size
        
        # Select transform based on train/test mode
        if train and augment:
            transform = get_transform(
                image_size=image_size,
                horizontal_flip=True,
            )
        else:
            transform = get_inference_transform(image_size=image_size)
        
        # Load CIFAR-10
        self.dataset = CIFAR10(
            root=str(self.root),
            train=train,
            download=download,
            transform=transform,
        )
        
        # Class names
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            (image, label) where image is (3, H, W) in [-1, 1]
        """
        return self.dataset[idx]


def get_cifar10_dataloader(
    root: str = "./data",
    train: bool = True,
    batch_size: int = 128,
    image_size: int = 32,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    pin_memory: bool = True,
    drop_last: bool = True,
    augment: bool = True,
) -> DataLoader:
    """
    Create CIFAR-10 DataLoader.
    
    Args:
        root: Root directory for dataset
        train: Load training set (True) or test set (False)
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        shuffle: Shuffle data (defaults to True for train, False for test)
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        augment: Apply data augmentation
    
    Returns:
        DataLoader for CIFAR-10
    """
    dataset = CIFAR10Dataset(
        root=root,
        train=train,
        image_size=image_size,
        augment=augment,
    )
    
    # Default shuffle: True for train, False for test
    if shuffle is None:
        shuffle = train
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
