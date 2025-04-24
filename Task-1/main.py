import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Must set this before importing numpy or torch

from models import TestDataset, extract_features
from ft_transformers import val_test_transforms, load_datasets
from submission import save_submission, evaluate_submission_accuracy
from fine_tune import fine_tune_model
from dummy_test import create_dummy_test_set
from perform_predictions import generate_predictions, predict_with_finetuned_model
from knn import train_knn, load_knn_model, evaluate_knn

from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import time
import torch


# %%
# --- Global Configuration ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KNN_MODEL_SAVE_PATH = "../models/cat_recognizer_knn.joblib" # Path to save the k-NN model

PERFORM_FINE_TUNING = False # Added: True/False
USE_FINE_TUNED_FOR_CLASSIFICATION = True
DUMMY_TEST_DIR = "../dummy_test_set" # Relative path for generated test set
DUMMY_SUBMISSION_FILE = "../dummy_submission.csv" # Path to save dummy submission
TEST_DIR = "../dataset/test" # Directory containing images to predict
NORMALIZE_FEATURES = True # Whether to normalize features before k-NN prediction
KNN_NEIGHBORS = 3
KNN_METRIC = 'cosine' # Options: 'cosine', 'euclidean'. Compare performance on validation set.
KNN_WEIGHTS = 'distance' # Options: 'uniform', 'distance'. 'distance' often performs well.
NORMALIZE_FEATURES = True # Often beneficial, especially with 'euclidean' distance. Try False too.
MODEL_NAME = 'resnet50' # Added: e.g., 'resnet50', 'efficientnet_b0', 'vit_b_16'
DUMMY_SAMPLE_SUBMISSION_FILE = "../dummy_sample_submission.csv" # Path to save dummy sample submission
FINE_TUNED_MODEL_SAVE_PATH = "../models/cat_recognizer_finetuned.pth" # Path to save fine-tuned model
BATCH_SIZE = 32  # Batch size for DataLoader
NUM_WORKERS = 4  # Number of workers for DataLoader


def load_model(num_classes, load_fine_tuned=False, model_path=None):
    """
    Loads a pre-trained or fine-tuned model based on global config.
    Handles different architectures and removes/replaces the final classifier layer.
    """
    print(f"\n--- Loading Model ({MODEL_NAME}) ---")
    model = None
    weights = None
    num_input_features = None

    # --- Load Pre-trained Model ---
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    num_input_features = model.fc.in_features
    model.fc = nn.Identity() # Remove classifier for feature extraction initially
    print(f"Loaded {MODEL_NAME} with ImageNet V2 weights.")

    if model is None:
        print("Model loading failed.")
        return None

    # --- Load Fine-tuned Weights if requested ---
    if load_fine_tuned:
        fine_tuned_path = model_path if model_path else FINE_TUNED_MODEL_SAVE_PATH
        if os.path.exists(fine_tuned_path):
            print(f"Loading fine-tuned weights from: {fine_tuned_path}")
            try:
                # Need to re-attach a temporary head matching the saved state's structure *before* loading
                if num_input_features is not None and num_classes is not None:
                     temp_classifier = nn.Linear(num_input_features, num_classes)
                     model.fc = temp_classifier
                state_dict = torch.load(fine_tuned_path, map_location=DEVICE)
                # Handle potential DataParallel prefix '_orig_mod..' if saved that way
                if list(state_dict.keys())[0].startswith('_orig_mod.'):
                     state_dict = {k.replace('_orig_mod.','') : v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                print("Fine-tuned weights loaded successfully.")

            except Exception as e:
                print(f"Error loading fine-tuned weights from {fine_tuned_path}: {e}")
                # Decide how to proceed: fall back to pre-trained or raise error?
                # For now, we continue with the potentially pre-trained weights loaded earlier.
                print("Warning: Falling back to base pre-trained weights.")
                # Ensure the head is removed if not doing direct classification
                model.fc = nn.Identity()
        else:
            print(f"Warning: Fine-tuned model path not found: {fine_tuned_path}. Using base pre-trained model.")
            # Ensure head is removed if feature extraction is the goal
            if not USE_FINE_TUNED_FOR_CLASSIFICATION:
                model.fc = nn.Identity()

    # --- Final Steps: Move to Device, Eval Mode, Compile ---
    model = model.to(DEVICE)
    model.eval() # Set to evaluation mode by default

    # Compile the model after moving to device and setting to eval (if not fine-tuning directly)
    # Skip compilation if we are about to fine-tune, compile after training setup instead.
    if not (PERFORM_FINE_TUNING and load_fine_tuned is False): # Compile if not actively fine-tuning now
        print("Attempting to compile model (PyTorch 2.0+)...")
        try:
            # Options: 'default', 'reduce-overhead', 'max-autotune'
            # 'reduce-overhead' is often good for inference/feature extraction
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed (requires PyTorch 2.0+ or compatible hardware/backend): {e}")

    return model, num_input_features


def main():
    """Main function to run the enhanced pipeline."""
    print("Starting Enhanced Cat Recognition Pipeline...")
    pipeline_start_time = time.time()


    # Generate/Verify Dummy Test Set (Optional, for local testing)
    # Set which test set to use (dummy or actual)
    use_dummy_test_set = True # Set to False to use the actual TEST_DIR
    if use_dummy_test_set:
        dummy_created_or_found = create_dummy_test_set()
        if not dummy_created_or_found:
            print("Failed to create or find dummy test set. Exiting.")
            return
        prediction_target_dir = DUMMY_TEST_DIR
        ground_truth_file = DUMMY_SAMPLE_SUBMISSION_FILE
    else:
        if not os.path.isdir(TEST_DIR) or not os.listdir(TEST_DIR):
             print(f"Actual test directory {TEST_DIR} is missing or empty. Cannot run predictions. Exiting.")
             return
        prediction_target_dir = TEST_DIR
        # No ground truth available for the actual test set usually
        ground_truth_file = None
        print(f"Using actual test directory: {TEST_DIR}")


    # 3. Load Datasets (Train/Val needed for fine-tuning or k-NN training/evaluation)
    train_loader, val_loader, idx_to_class, num_classes  = load_datasets()
    if train_loader is None or val_loader is None or idx_to_class is None:
        print("Failed to load datasets. Exiting.")
        return

    # 4. Load Model (Pre-trained or Fine-tuned)
    model = None
    num_input_features = None # Needed for adding head during fine-tuning

    if PERFORM_FINE_TUNING:
        # Load the base model architecture first, but don't load fine-tuned weights yet
        print("Loading base model structure for fine-tuning...")
        base_model, num_input_features = load_model(num_classes, load_fine_tuned=True)
        if base_model is None or num_input_features is None:
            print("Failed to load base model structure for fine-tuning. Exiting.")
            return

        # Perform the fine-tuning process
        # This function adds the head, trains, and saves the best model state
        tuned_model = fine_tune_model(base_model, num_input_features, train_loader, val_loader, num_classes, idx_to_class)

        if tuned_model is None:
            print("Fine-tuning failed. Exiting.")
            return

        # After tuning, decide whether to use it for direct classification or feature extraction
        if USE_FINE_TUNED_FOR_CLASSIFICATION:
            print("Using fine-tuned model for direct classification.")
            model = tuned_model # Keep the model with the head
            # Set knn_classifier to None to signal bypassing k-NN steps
            knn_classifier = None
        else:
            print("Using fine-tuned model as feature extractor for k-NN.")
            # Load the *best saved* fine-tuned weights, ensuring the head is removed.
            # The `load_model` function handles removing the head based on USE_FINE_TUNED_FOR_CLASSIFICATION=False
            model, _ = load_model(num_classes, load_fine_tuned=True)
            if model is None:
                 print("Failed to load the saved fine-tuned model weights for feature extraction. Exiting.")
                 return
            # Proceed to k-NN training/loading below...
            knn_classifier = None # Will be trained or loaded next

    else:
        # Not fine-tuning, load pre-trained model directly for feature extraction
        print("Loading pre-trained model for feature extraction (no fine-tuning).")
        model, _ = load_model(num_classes, load_fine_tuned=False) # Head is removed by default here
        if model is None:
            print("Failed to load pre-trained model. Exiting.")
            return
        knn_classifier = None # Will be trained or loaded next


    # --- k-NN Workflow (Execute if not doing direct classification) ---
    submission_df = None
    if not (PERFORM_FINE_TUNING and USE_FINE_TUNED_FOR_CLASSIFICATION):
        print("\n--- Entering k-NN Workflow ---")
        knn_classifier = None
        # Check if a pre-trained k-NN model exists and matches config
        if os.path.exists(KNN_MODEL_SAVE_PATH):
            print(f"Found existing k-NN model at {KNN_MODEL_SAVE_PATH}.")
            # Optional: Add check here to see if KNN params in config match the loaded model
            # If they don't match, you might want to force retraining.
            knn_classifier = load_knn_model()

        # Train k-NN if not loaded or if forced
        if knn_classifier is None: # Or add a FORCE_KNN_RETRAIN flag
            print("Training new k-NN model...")
            # Extract Training Features
            train_features, train_labels = extract_features(train_loader, model, description="training set", normalize_feats=NORMALIZE_FEATURES)
            if train_features is None or len(train_features) == 0:
                print("Failed to extract training features. Exiting.")
                return

            # Train k-NN Classifier
            knn_classifier = train_knn(train_features, train_labels)
            if knn_classifier is None:
                print("Failed to train k-NN classifier. Exiting.")
                return

            # Evaluate on Validation Set (Important for hyperparameter tuning)
            print("Extracting validation features for evaluation...")
            val_features, val_labels = extract_features(val_loader, model, description="validation set", normalize_feats=NORMALIZE_FEATURES)
            if val_features is not None and len(val_features) > 0:
                 evaluate_knn(knn_classifier, val_features, val_labels, idx_to_class)
            else:
                 print("Skipping k-NN evaluation due to issues extracting validation features.")

        else:
             print("Using pre-loaded k-NN model.")
             knn_classifier = load_knn_model()
             # Optionally, still run evaluation on validation set with the loaded model
             # print("Extracting validation features for evaluation with loaded k-NN...")
             # val_features, val_labels = extract_features(val_loader, model, description="validation set", normalize_feats=NORMALIZE_FEATURES)
             # if val_features is not None:
             #      evaluate_knn(knn_classifier, val_features, val_labels, idx_to_class)

        # Generate Predictions using k-NN
        submission_df = generate_predictions(knn_classifier, model, idx_to_class, test_dir=prediction_target_dir)

    # --- Direct Classification Workflow ---
    elif PERFORM_FINE_TUNING and USE_FINE_TUNED_FOR_CLASSIFICATION:
         print("\n--- Entering Direct Classification Workflow ---")
         # Load the test dataset
         test_dataset = TestDataset(prediction_target_dir, transform=val_test_transforms)
         if len(test_dataset) > 0:
             test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
             # Generate predictions using the fine-tuned model directly
             submission_df = predict_with_finetuned_model(model, test_loader, idx_to_class)
         else:
             print(f"Test dataset at '{prediction_target_dir}' is empty or not found. Cannot generate predictions.")


    # 7. Save Submission
    if submission_df is not None:
        save_submission(submission_df, DUMMY_SUBMISSION_FILE)
        print(f"Submission saved to: {DUMMY_SUBMISSION_FILE}")
    else:
        print("No submission DataFrame generated.")

    # 8. Evaluate Submission if Ground Truth is Available
    if ground_truth_file and os.path.exists(ground_truth_file) and submission_df is not None:
         evaluate_submission_accuracy(DUMMY_SUBMISSION_FILE, ground_truth_file)
    elif use_dummy_test_set:
         print(f"Could not evaluate submission because ground truth file '{ground_truth_file}' was not found or submission failed.")
    else:
         print("Skipping evaluation as no ground truth file is specified for the actual test set.")


    pipeline_end_time = time.time()
    print("\nPipeline finished.")
    print(f"Total execution time: {pipeline_end_time - pipeline_start_time:.2f} seconds")


if __name__ == '__main__':
    main()