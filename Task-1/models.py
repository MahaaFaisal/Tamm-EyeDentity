import os
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from sklearn.preprocessing import normalize
from torch.cuda.amp import autocast
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

# --- Custom Test Dataset Class --- (Remains largely the same)
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if not os.path.isdir(root_dir):
            print(f"Warning: Test directory '{root_dir}' not found. TestDataset will be empty.")
            self.image_filenames = []
        else:
            self.image_filenames = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Found {len(self.image_filenames)} image files in test directory: {root_dir}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, img_name
        except UnidentifiedImageError:
             print(f"\nWarning: Corrupted or unidentified image {img_path}. Returning dummy tensor.")
             return torch.zeros((3, IMG_SIZE, IMG_SIZE)), img_name # Return dummy tensor
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor and the filename
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), img_name
        

def extract_features(dataloader, model, description="data", normalize_feats=False):
    """
    Extracts features and labels/filenames. Optionally normalizes features. Uses global DEVICE.
    """

    if dataloader is None:
        print(f"Dataloader for {description} is None, skipping feature extraction.")
        return None, None

    features_list = []
    meta_list = [] # Holds labels (train/val) or filenames (test)
    is_test_loader = isinstance(dataloader.dataset, TestDataset)

    model.eval() # Ensure model is in eval mode
    num_batches = len(dataloader)
    print(f"Starting feature extraction for {description} ({num_batches} batches)...")

    with torch.no_grad(): # Essential for inference
        for i, batch_data in enumerate(dataloader):
            if is_test_loader:
                inputs, meta_batch = batch_data # TestDataset yields (image, filename)
                # Handle potential dummy tensors from TestDataset error loading
                valid_indices = [j for j, img in enumerate(inputs) if img.nelement() > 0 and not torch.equal(img, torch.zeros_like(img))]
                if len(valid_indices) < len(inputs):
                    print(f"Warning: Batch {i+1}/{num_batches} contains {len(inputs) - len(valid_indices)} dummy tensors due to loading errors.")
                    if not valid_indices:
                        print(f"Skipping Batch {i+1} entirely as it contains only dummy tensors.")
                        continue # Skip batch if all are dummy
                    inputs = inputs[valid_indices]
                    # Filter meta_batch accordingly - needs careful indexing if meta_batch is not a tensor
                    meta_batch = [meta_batch[j] for j in valid_indices]


            else:
                inputs, meta_batch = batch_data # ImageFolder yields (image, label_idx)
                meta_batch = meta_batch.cpu().tolist() # Convert tensor labels to list of ints

            inputs = inputs.to(DEVICE)

            # Use autocast for potential mixed-precision inference speedup
            with autocast(enabled=(DEVICE.type == 'cuda')): # Only autocast on CUDA
                outputs = model(inputs)

            # Outputs might be FP16 if autocast is used on GPU, convert back to FP32 before CPU/numpy
            features_list.append(outputs.float().cpu().numpy())
            meta_list.extend(meta_batch) # Add labels or filenames

            print(f" Batch {i+1}/{num_batches} processed.", end='\r')

    print(f"\nFinished feature extraction for {description}.")

    if not features_list:
        print(f"No features were extracted for {description}.")
        return np.array([]), []

    features = np.concatenate(features_list, axis=0)
    print(f" Extracted raw features shape: {features.shape}")

    # --- Optional Feature Normalization ---
    if normalize_feats:
        print(" Normalizing features (L2 norm)...")
        features = normalize(features, norm='l2', axis=1)
        print(f" Normalized features shape: {features.shape}")

    return features, meta_list