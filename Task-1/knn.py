import os
import time
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

KNN_MODEL_SAVE_PATH = "models/cat_recognizer_knn.joblib" # Path to save the k-NN model
KNN_METRIC = 'cosine' # Options: 'cosine', 'euclidean'. Compare performance on validation set.
KNN_WEIGHTS = 'distance' # Options: 'uniform', 'distance'. 'distance' often performs well.
KNN_MODEL_SAVE_PATH = "models/cat_recognizer_knn.joblib" # Path to save the k-NN model

def train_knn(train_features, train_labels):
    """Trains and saves the k-NN classifier using global config for hyperparameters."""

    print(f"\n--- Training k-NN Classifier ---")
    knn_save_path = KNN_MODEL_SAVE_PATH
    k = 3
    metric = KNN_METRIC
    weights = KNN_WEIGHTS
    knn_classifier = None

    if train_features is None or train_labels is None or len(train_features) == 0:
        print("Error: Training features or labels are missing. Cannot train k-NN.")
        return None

    if len(train_features) != len(train_labels):
        print(f"Error: Mismatch between feature count ({len(train_features)}) and label count ({len(train_labels)}).")
        return None

    print(f"Training k-NN with k={k}, metric='{metric}', weights='{weights}' on {len(train_features)} samples...")
    start_time = time.time()

    try:
        # n_jobs=-1 uses all available CPU cores
        knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1)
        knn_classifier.fit(train_features, train_labels)
        end_time = time.time()
        print(f"k-NN training complete. Time taken: {end_time - start_time:.2f} seconds")

        print(f"Saving k-NN model to: {knn_save_path}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(knn_save_path), exist_ok=True)
        joblib.dump(knn_classifier, knn_save_path)
        print("k-NN model saved successfully.")

    except Exception as e:
        print(f"An error occurred during k-NN training or saving: {e}")
        return None

    return knn_classifier

# %% [markdown]
# ### 7.5. k-NN Evaluation (New Function for Hyperparameter Tuning)

# %%
def evaluate_knn(knn_classifier, val_features, val_labels, idx_to_class):
    """Evaluates the trained k-NN classifier on the validation set."""

    print(f"\n--- Evaluating k-NN on Validation Set ---")
    if knn_classifier is None or val_features is None or val_labels is None or len(val_features) == 0:
        print("Error: Missing k-NN model, validation features, or labels for evaluation.")
        return

    if len(val_features) != len(val_labels):
        print(f"Error: Mismatch validation feature count ({len(val_features)}) vs label count ({len(val_labels)}).")
        return

    k = knn_classifier.n_neighbors # Get k from the trained classifier itself
    print(f"Predicting top-{k} labels for {len(val_features)} validation samples...")
    start_time = time.time()

    try:
        # Get probabilities to find top-k predictions
        val_probabilities = knn_classifier.predict_proba(val_features)
        # Get the indices of the top k neighbors for each validation sample
        # argsort sorts in ascending order, so take the last k indices
        top_k_indices = np.argsort(val_probabilities, axis=1)[:, -k:]

        end_time = time.time()
        print(f"Validation prediction complete. Time taken: {end_time - start_time:.2f} seconds")

        # Map predicted indices back to original class indices used during training
        predicted_class_indices_topk = knn_classifier.classes_[top_k_indices]

        # Check if the true label is within the top-k predictions
        hits = 0
        for i, true_label_idx in enumerate(val_labels):
            # Get the top k predicted class indices for the i-th sample (in descending order of probability)
            predicted_indices_for_sample = predicted_class_indices_topk[i][::-1] # Reverse to get highest prob first
            if true_label_idx in predicted_indices_for_sample:
                hits += 1

        accuracy_topk = hits / len(val_labels)
        print(f"Validation Top-{k} Accuracy: {accuracy_topk:.4f} ({hits}/{len(val_labels)})")

        # Optional: Calculate Top-1 accuracy as well
        top_1_indices = top_k_indices[:, -1] # Index of the single best prediction
        predicted_class_indices_top1 = knn_classifier.classes_[top_1_indices]
        accuracy_top1 = accuracy_score(val_labels, predicted_class_indices_top1)
        print(f"Validation Top-1 Accuracy: {accuracy_top1:.4f}")

    except Exception as e:
        print(f"Error during k-NN validation evaluation: {e}")

    # --- How to use this for tuning ---
    # 1. Run the main script with different `3` values (e.g., 1, 3, 5, 7, 9...).
    # 2. Run with `KNN_METRIC='cosine'` vs `'euclidean'`.
    # 3. Run with `KNN_WEIGHTS='uniform'` vs `'distance'`.
    # 4. Run with `NORMALIZE_FEATURES=True` vs `False`.
    # 5. Choose the combination of hyperparameters that yields the best validation accuracy (Top-k or Top-1).


# %% [markdown]
# ### 8. Prediction Generation (Adjusted for Normalization and Clarity)

# %%



# %% [markdown]
# ### 9. Save Submission (No change needed)

# %%

def load_knn_model():
    """Loads a pre-trained k-NN model."""
    global KNN_MODEL_SAVE_PATH
    try:
        if not os.path.exists(KNN_MODEL_SAVE_PATH):
            print(f"Error: k-NN model file not found at {KNN_MODEL_SAVE_PATH}")
            return None
        knn_classifier = joblib.load(KNN_MODEL_SAVE_PATH)
        print(f"\n--- k-NN Model Loaded ---")
        print(f"k-NN model loaded from: {KNN_MODEL_SAVE_PATH}")
        # You might want to verify hyperparameters match current config
        print(f" Loaded model params: k={knn_classifier.n_neighbors}, metric='{knn_classifier.metric}', weights='{knn_classifier.weights}'")
        return knn_classifier
    except Exception as e:
        print(f"Error loading k-NN model from {KNN_MODEL_SAVE_PATH}: {e}")
        return None