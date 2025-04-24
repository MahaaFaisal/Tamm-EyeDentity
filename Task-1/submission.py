import os
import pandas as pd

def evaluate_submission_accuracy(submission_file, ground_truth_file):
    """Calculates Top-K accuracy using the generated submission and ground truth."""

    print(f"\n--- Evaluating Submission Accuracy ---")
    try:
        submission_df = pd.read_csv(submission_file)
        ground_truth_df = pd.read_csv(ground_truth_file)
    except FileNotFoundError as e:
        print(f"Error: Cannot load files for evaluation: {e}")
        return
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Merge based on filename to align predictions and true labels
    eval_df = pd.merge(submission_df, ground_truth_df, on='filename', how='inner')

    if len(eval_df) == 0:
        print("Error: No matching filenames found between submission and ground truth. Cannot evaluate.")
        return

    if len(eval_df) != len(submission_df) or len(eval_df) != len(ground_truth_df):
        print(f"Warning: Evaluation set size ({len(eval_df)}) differs from submission ({len(submission_df)}) or ground truth ({len(ground_truth_df)}). Evaluation based on intersection.")


    hits = 0
    label_cols = [f'label_{i+1}' for i in range(3)] # Generate label column names based on k

    for index, row in eval_df.iterrows():
        true_label = row['label'] # Assumes ground truth column is named 'label'
        predicted_labels = [row[col] for col in label_cols if col in row and pd.notna(row[col])] # Get top-k predictions

        if true_label in predicted_labels:
            hits += 1
        # Optional: Print mismatch details for debugging
        # else:
            # print(f" Miss: {row['filename']} - True: {true_label}, Predicted: {predicted_labels}")


    accuracy = hits / len(eval_df) if len(eval_df) > 0 else 0
    print(f"Evaluation Accuracy (Top-{3}): {accuracy:.4f} ({hits}/{len(eval_df)})")
    return accuracy

def save_submission(df, SUBMISSION_FILE):
    """Saves the prediction DataFrame to a CSV file. Uses global SUBMISSION_FILE."""

    if df is None:
        print("No submission DataFrame provided to save.")
        return

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)
        df.to_csv(SUBMISSION_FILE, index=False)
        print(f"\n--- Submission Saved ---")
        print(f"Submission file successfully saved to: {SUBMISSION_FILE}")
    except Exception as e:
        print(f"Error saving submission file to {SUBMISSION_FILE}: {e}")