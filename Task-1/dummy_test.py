import os
import random
import shutil
import pandas as pd
from PIL import Image, UnidentifiedImageError

# %% [markdown]
# ### Dummy Test Set Generation (Minor adjustments for clarity)

TRAIN_DIR = '../dataset/train'  # Path to training data
VAL_DIR = '../dataset/val'  # Path to validation data
DUMMY_TEST_DIR = '../dataset/dummy_test'  # Path to dummy test data
NUM_DUMMY_TEST_IMAGES = 1000  # Number of dummy test images to generate
DUMMY_SAMPLE_SUBMISSION_FILE = '../dataset/dummy_test/sample_submission.csv'  # Dummy submission file path
FORCE_DUMMY_REGENERATE = False  # Force regeneration of dummy test set
IMG_SIZE = 256  # Image size for resizing

# %%
def create_dummy_test_set():
    """
    Generates a dummy test set by augmenting images from the validation set.
    Uses global configuration variables.
    """

    print(f"\n--- Generating Dummy Test Set ---")
    # Use TRAIN_DIR as source for more variety if VAL_DIR is small
    source_dir = TRAIN_DIR # Or VAL_DIR if preferred
    target_dir = DUMMY_TEST_DIR
    num_images = NUM_DUMMY_TEST_IMAGES
    dummy_sample_submission_file = DUMMY_SAMPLE_SUBMISSION_FILE

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory for dummy data '{source_dir}' not found. Skipping generation.")
        return False

    # Check if dummy test files exist and handle regeneration flag
    if os.path.exists(target_dir):
        if FORCE_DUMMY_REGENERATE:
            print(f"FORCE_DUMMY_REGENERATE is True. Removing existing dummy test directory: {target_dir}")
            shutil.rmtree(target_dir)
        else:
             print(f"Dummy test directory '{target_dir}' already exists and FORCE_DUMMY_REGENERATE is False.")
             # Check if submission file also exists, create if not
             if not os.path.exists(dummy_sample_submission_file):
                 print("Generating missing dummy sample submission file...")
                 image_files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
                 if image_files:
                      # Recreate from image filenames if labels aren't easily available
                      dummy_sub_df = pd.DataFrame({'filename': image_files})
                      # Ideally, load the previously generated labels if they were saved.
                      # For simplicity here, just creating the structure.
                      dummy_sub_df['label'] = 'unknown' # Placeholder
                      dummy_sub_df.to_csv(dummy_sample_submission_file, index=False)
                      print(f"Dummy sample submission created at {dummy_sample_submission_file}")
                 else:
                      print("Dummy directory is empty, cannot create sample submission.")
             else:
                 print("Dummy sample submission file already exists. Skipping generation.")
             return True # Skip generation if directory exists and no force regenerate

    os.makedirs(target_dir, exist_ok=True)
    print(f"Creating dummy test set in: {target_dir}")

    source_image_paths = []
    source_labels = [] # Store original labels
    for cat_id_dir in os.listdir(source_dir):
        cat_path = os.path.join(source_dir, cat_id_dir)
        if os.path.isdir(cat_path):
            for img_file in os.listdir(cat_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_image_paths.append(os.path.join(cat_path, img_file))
                    source_labels.append(cat_id_dir) # Store the class name (folder name)

    if not source_image_paths:
        print(f"Error: No images found in the source directory '{source_dir}'. Cannot generate dummy data.")
        return False

    num_available = len(source_image_paths)
    indices_to_sample = list(range(num_available))

    generated_filenames = []
    generated_labels = [] # Store corresponding true labels for the dummy set

    print(f"Generating {num_images} augmented images...")
    for i in range(num_images):
        try:
            # Sample with replacement if needed
            chosen_idx = random.choice(indices_to_sample)
            img_path = source_image_paths[chosen_idx]
            original_label = source_labels[chosen_idx]

            img = Image.open(img_path).convert('RGB')
            img_aug = dummy_generation_transforms(img)

            # Ensure unique filenames even if source image repeats
            target_filename = f"dummy_test_{i:05d}_{original_label}.png" # Include label hint for debugging
            target_path = os.path.join(target_dir, target_filename)
            img_aug.save(target_path)
            generated_filenames.append(target_filename)
            generated_labels.append(original_label) # Save the true label
            print(f"  Generated: {target_filename} (from {os.path.basename(img_path)})", end='\r')
        except UnidentifiedImageError:
            print(f"\nWarning: Skipping corrupted or unidentified image: {img_path}")
        except Exception as e:
            print(f"\nError processing image {img_path}: {e}")

    print(f"\nSuccessfully generated {len(generated_filenames)} dummy test images.")

    if generated_filenames:
        # Create the dummy submission file with TRUE labels for evaluation
        dummy_sub_df = pd.DataFrame({'filename': generated_filenames, 'label': generated_labels})
        dummy_sub_df.to_csv(dummy_sample_submission_file, index=False)
        print(f"Dummy sample submission file (with true labels) created at: {dummy_sample_submission_file}")
    else:
        print("No images were generated, skipping dummy sample submission creation.")
        return False

    return True