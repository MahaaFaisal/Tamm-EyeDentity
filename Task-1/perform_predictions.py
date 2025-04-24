import torch
import pandas as pd
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import numpy as np
import joblib
import time
import os
from ft_transformers import val_test_transforms
from models import TestDataset, extract_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # Batch size for DataLoader
NUM_WORKERS = 4  # Number of workers for DataLoader
KNN_NEIGHBORS = 3  # Number of neighbors
KNN_MODEL_SAVE_PATH = "../models/cat_recognizer_knn.joblib" # Path to save the k-NN model
NORMALIZE_FEATURES = True  # Whether to normalize features before k-NN prediction

def generate_predictions(knn_classifier, model, idx_to_class, test_dir):
    """
    Generates predictions for the test set using k-NN.
    Uses global config for k-NN parameters and normalization.
    """
    print(f"\n--- Generating Predictions ---")
    # test_dir is now passed as an argument (e.g., DUMMY_TEST_DIR or TEST_DIR)
    pin_memory = (DEVICE != torch.device('cpu'))

    # --- Load k-NN model if not provided ---
    if knn_classifier is None:
        print("k-NN classifier not provided.")
        if os.path.exists(KNN_MODEL_SAVE_PATH):
            try:
                print(f"Attempting to load k-NN model from {KNN_MODEL_SAVE_PATH}")
                knn_classifier = joblib.load(KNN_MODEL_SAVE_PATH)
                print("Loaded k-NN model successfully.")
                # Update global KNN_NEIGHBORS based on loaded model if needed, though config should match ideally
                # KNN_NEIGHBORS = knn_classifier.n_neighbors
            except Exception as e:
                print(f"Error loading k-NN model: {e}. Aborting prediction.")
                return None
        else:
            print("k-NN model file not found. Aborting prediction.")
            return None

    if model is None:
        print("Error: Feature extraction model is not loaded. Aborting prediction.")
        return None
    if idx_to_class is None:
        print("Error: Class mapping (idx_to_class) is not available. Aborting prediction.")
        return None

    # --- Load Test Data ---
    print(f"Loading test data from: {test_dir}")
    test_dataset = TestDataset(test_dir, transform=val_test_transforms)
    if len(test_dataset) == 0:
        print(f"Test dataset at '{test_dir}' is empty or not found. Cannot generate predictions.")
        return None
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)

    # --- Extract Test Features (apply normalization if used during training) ---
    test_features, test_filenames = extract_features(test_loader, model, description="test set", normalize_feats=NORMALIZE_FEATURES)

    if test_features is None or len(test_features) == 0 or len(test_features) != len(test_filenames):
         print(f"Error in test feature extraction (Features: {test_features.shape if test_features is not None else 'None'}, Filenames: {len(test_filenames)}). Aborting prediction.")

         return None


    # --- Predict with k-NN ---
    k = knn_classifier.n_neighbors # Use k from the actual model
    print(f"Predicting top {k} labels using k-NN for {len(test_features)} test images...")
    start_time = time.time()
    try:
        # Predict probabilities to get ranking
        test_probabilities = knn_classifier.predict_proba(test_features)
        # Get indices of the top k neighbors (sorted ascending by probability, take last k)
        top_k_indices = np.argsort(test_probabilities, axis=1)[:, -k:]
        end_time = time.time()
        print(f"Prediction complete. Time taken: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"Error during k-NN prediction: {e}")
        return None

    # --- Map Indices to Labels ---
    print("Mapping prediction indices to class labels...")
    top_k_labels_list = []
    num_classes_knn = len(knn_classifier.classes_)
    # Get the actual class indices corresponding to the k-NN internal indices
    knn_class_mapping = knn_classifier.classes_

    for i, row_indices in enumerate(top_k_indices):
        # print(f" Processing row {i+1}/{len(top_k_indices)}...", end='\r') # Can be verbose
        labels = []
        # Iterate through the indices in reverse order (highest probability first)
        for idx in reversed(row_indices):
            if 0 <= idx < num_classes_knn:
                original_class_index = knn_class_mapping[idx] # Map internal k-NN index to original dataset index
                label_str = idx_to_class.get(original_class_index, f"UnknownIdx_{original_class_index}")
                labels.append(label_str)
            else:
                # This should ideally not happen if k-NN is trained correctly
                print(f"\nWarning: Predicted index {idx} out of bounds for k-NN classes (size {num_classes_knn}) in sample {i}.")
                labels.append("OOB_IDX") # Out of Bounds Index
        top_k_labels_list.append(labels)
    # print("\nLabel mapping finished.") # Clear the carriage return line

    if len(test_filenames) != len(top_k_labels_list):
        print(f"Error: Mismatch filename count ({len(test_filenames)}) vs prediction count ({len(top_k_labels_list)}).")
        return None

    # --- Create Submission DataFrame ---
    # Ensure exactly k labels per row, padding with a placeholder if fewer unique predictions found
    # (though predict_proba usually gives k distinct probabilities/indices unless duplicates in training data are very close)
    padded_labels = []
    for row in top_k_labels_list:
         padded_row = row + ['None'] * (k - len(row))
         padded_labels.append(padded_row[:k]) # Ensure exactly k

    # Create dictionary for DataFrame columns dynamically based on k
    submission_data = {'filename': test_filenames}
    for i in range(k):
         submission_data[f'label_{i+1}'] = [labels[i] for labels in padded_labels]

    submission_df = pd.DataFrame(submission_data)

    print("\nSubmission DataFrame created:")
    print(submission_df.head())
    return submission_df



def predict_with_finetuned_model(model, test_loader, idx_to_class):
     """
     Generates predictions using the fine-tuned model's classifier head.
     (Requires implementation similar to generate_predictions but using model output directly)
     """
     print("\n--- Generating Predictions using Fine-tuned Classifier ---")
     model.eval()
     model = model.to(DEVICE)
     pin_memory = (DEVICE.type == 'cuda')

     test_filenames = []
     all_predictions = [] # Store lists of top-k predicted labels

     with torch.no_grad():
          num_batches = len(test_loader)
          for i, (inputs, filenames) in enumerate(test_loader):
               print(f" Processing test batch {i+1}/{num_batches}...", end='\r')
               inputs = inputs.to(DEVICE)
               # Handle dummy tensors if TestDataset returns them
               valid_indices = [j for j, img in enumerate(inputs) if img.nelement() > 0 and not torch.equal(img, torch.zeros_like(img))]
               if len(valid_indices) < len(inputs):
                  if not valid_indices: continue # Skip batch if all dummy
                  inputs = inputs[valid_indices]
                  filenames = [filenames[j] for j in valid_indices]

               with autocast(enabled=(DEVICE.type == 'cuda')):
                    outputs = model(inputs) # Get raw logits from the classifier head
                    probabilities = torch.softmax(outputs, dim=1) # Convert to probabilities

               # Get top-k predictions
               print(f"Getting top-{KNN_NEIGHBORS} predictions...", end='\r')
               top_p, top_class_indices = probabilities.topk(KNN_NEIGHBORS, dim=1, largest=True, sorted=True)
               top_class_indices = top_class_indices.cpu().numpy()

               # Map indices to labels
               batch_predictions = []
               for sample_indices in top_class_indices:
                    sample_labels = [idx_to_class.get(idx, f"UnknownIdx_{idx}") for idx in sample_indices]
                    batch_predictions.append(sample_labels)

               all_predictions.extend(batch_predictions)
               test_filenames.extend(filenames)

     print("\nFinished predicting with fine-tuned model.")

     if len(test_filenames) != len(all_predictions):
          print(f"Error: Mismatch filename count ({len(test_filenames)}) vs prediction count ({len(all_predictions)}).")
          return None

     # --- Create Submission DataFrame ---
     k = KNN_NEIGHBORS
     submission_data = {'filename': test_filenames}
     for i in range(k):
         submission_data[f'label_{i+1}'] = [labels[i] if i < len(labels) else 'None' for labels in all_predictions]

     submission_df = pd.DataFrame(submission_data)
     print("\nSubmission DataFrame created:")
     print(submission_df.head())
     return submission_df