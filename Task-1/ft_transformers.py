from torchvision import transforms
from torchvision.transforms import InterpolationMode
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

IMAGE_SIZE = 256  # Image size for resizing

TRAIN_DIR = '../dataset/train'  # Path to training data
VAL_DIR = '../dataset/val'  # Path to validation data
BATCH_SIZE = 32  # Batch size for DataLoader
NUM_WORKERS = 4  # Number of workers for DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for training

# --- Define Transforms --- (Using global IMG_SIZE and enhanced augmentations)
# Enhanced transforms with augmentation
# Using RandomResizedCrop for better training robustness
train_transforms = transforms.Compose([
    # RandomResizedCrop is often better than Resize for training robustness
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5), # Increased rotation slightly
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Small random translation
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Slightly stronger jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Keep validation/test transforms minimal
val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Basic augmentations for dummy generation
dummy_generation_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    # No ToTensor or Normalize needed here, saving PIL Image
])


def load_datasets():
    """Loads train/val datasets and creates dataloaders. Uses global config."""

    print(f"\n--- Loading Datasets ---")
    pin_memory = (DEVICE != torch.device('cpu'))
    train_loader, val_loader, idx_to_class = None, None, None

    try:
        print(f"Loading training data from: {TRAIN_DIR}")
        # Check if TRAIN_DIR exists and is not empty
        if not os.path.isdir(TRAIN_DIR) or not os.listdir(TRAIN_DIR):
             raise FileNotFoundError(f"Training directory '{TRAIN_DIR}' not found or is empty.")
        train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)

        print(f"Loading validation data from: {VAL_DIR}")
         # Check if VAL_DIR exists and is not empty
        if not os.path.isdir(VAL_DIR) or not os.listdir(VAL_DIR):
             raise FileNotFoundError(f"Validation directory '{VAL_DIR}' not found or is empty.")
        val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_test_transforms)

        class_to_idx = train_dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx) # Set global NUM_CLASSES
        print(f"Found {num_classes} unique cat IDs (classes).")
        # print(f"Class mapping (idx_to_class): {idx_to_class}") # Optional: print mapping

        # Address potential class imbalance (informational)
        # TODO: Consider strategies like oversampling/undersampling if fine-tuning,
        # or weighted loss functions if directly classifying after fine-tuning.
        print("Class distribution in training set:")
        print(pd.Series(train_dataset.targets).value_counts())


        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory, drop_last=True) # drop_last can help with batch norm stability
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)
        print("Train and Validation DataLoaders created.")

    except FileNotFoundError as e:
        print(f"Error: Data directory not found - {e}. Cannot create DataLoaders.")
        return None, None, None # Return tuple
    except Exception as e:
        print(f"An unexpected error occurred during dataset loading: {e}")
        return None, None, None # Return tuple

    return train_loader, val_loader, idx_to_class, num_classes