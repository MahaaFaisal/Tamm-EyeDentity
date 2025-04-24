# Cat Breed Recognition Pipeline

This project implements a configurable pipeline for cat breed recognition using deep learning models (specifically Convolutional Neural Networks) and optionally K-Nearest Neighbors (k-NN). It leverages pre-trained models from `torchvision`, allows for fine-tuning on a custom dataset, and provides options for classification either directly with the fine-tuned model or using a k-NN classifier on extracted features.

## Features

* **Pre-trained Models:** Utilizes models like ResNet50 pre-trained on ImageNet. (Easily adaptable for other `torchvision` models).
* **Fine-Tuning:** Option to fine-tune the selected CNN model on your specific cat breed dataset for improved performance.
* **Feature Extraction:** Extract deep features from images using the (potentially fine-tuned) CNN.
* **k-NN Classification:** Option to train and use a k-NN classifier on the extracted features. Useful when fine-tuning is not performed or as an alternative classification head.
* **Direct Classification:** Option to use the fine-tuned CNN model directly for end-to-end classification.
* **Configurable Workflow:** Control whether to perform fine-tuning, use the fine-tuned model directly, or employ the k-NN approach via simple boolean flags.
* **Hyperparameter Tuning (k-NN):** Configure k-NN parameters like the number of neighbors, distance metric (`cosine`, `euclidean`), and weighting scheme (`uniform`, `distance`).
* **Feature Normalization:** Option to normalize extracted features before feeding them to the k-NN classifier.
* **Model Persistence:** Saves and loads fine-tuned model weights (`.pth`) and trained k-NN models (`.joblib`).
* **Dummy Test Set:** Includes functionality to generate a dummy test set from the validation set for local testing and debugging.
* **Submission Generation:** Creates a prediction file (`.csv`) in a standard format (e.g., for competitions or evaluation).
* **Evaluation:** Calculates accuracy if a ground truth file for the test set (e.g., for the dummy set) is available.
* **Performance:** Uses `torch.compile` (if available) for potential speedups and efficient `DataLoader` settings.

## Project Structure

.
├── dataset/                # Root directory for image data
│   ├── train/              # Training images (organized by class folders)
│   │   ├── breed_A/
│   │   │   ├── img1.jpg
│   │   │   └── ...
│   │   └── breed_B/
│   │       └── ...
│   ├── val/                # Validation images (organized by class folders)
│   │   └── ...
│   └── test/               # Test images (flat structure, names used as IDs)
│       ├── test_img1.jpg
│       └── test_img2.jpg
│           └── ...
├── models/                 # Directory to save trained models
│   ├── cat_recognizer_finetuned.pth # Example saved fine-tuned model
│   └── cat_recognizer_knn.joblib    # Example saved k-NN model
├── src/                    # Source code directory (assuming modules are here)
│   ├── main.py             # Main pipeline script (the provided code)
│   ├── models.py           # TestDataset, extract_features
│   ├── ft_transformers.py  # Transforms, load_datasets
│   ├── submission.py       # save_submission, evaluate_submission_accuracy
│   ├── fine_tune.py        # fine_tune_model
│   ├── dummy_test.py       # create_dummy_test_set
│   ├── perform_predictions.py # generate_predictions, predict_with_finetuned_model
│   ├── knn.py              # train_knn, load_knn_model, evaluate_knn
│   └── requirements.txt    # Project dependencies
├── dummy_test_set/         # Auto-generated dummy test images (if enabled)
├── dummy_submission.csv    # Predictions for the dummy test set
└── dummy_sample_submission.csv # Ground truth for the dummy test set
└── README.md               # This file


*(Note: Adapt the `src/` directory structure if your files are organized differently)*

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **Install dependencies:**
    * Ensure you have Python 3.8+ installed.
    * Install PyTorch matching your CUDA version (if using GPU) or the CPU version. See [PyTorch installation guide](https://pytorch.org/get-started/locally/).
    * Install other required packages:
        ```bash
        pip install -r src/requirements.txt
        ```
        *(You'll need to create `src/requirements.txt` based on the imports)*

    **Example `requirements.txt`:**
    ```txt
    torch
    torchvision
    scikit-learn
    joblib
    numpy
    pandas
    tqdm # Often used in helper functions like extract_features
    ```

4.  **Prepare Dataset:**
    * Place your training, validation, and test images into the `dataset/` directory following the structure outlined above.
    * Training and validation images should be inside subdirectories named after their respective classes (breeds).
    * Test images should be directly inside the `test/` directory.

## Configuration

Modify the global configuration variables at the top of `main.py` (or your main script file) to control the pipeline's behavior:

* `DEVICE`: Automatically set to CUDA if available, otherwise CPU.
* `KNN_MODEL_SAVE_PATH`: Path to save/load the trained k-NN model.
* `PERFORM_FINE_TUNING`: `True` to fine-tune the base CNN model, `False` to use the pre-trained weights directly (for feature extraction).
* `USE_FINE_TUNED_FOR_CLASSIFICATION`: (Only relevant if `PERFORM_FINE_TUNING` is `True`).
    * `True`: Use the fine-tuned model's final layer for direct classification.
    * `False`: Use the fine-tuned model as a feature extractor and then use k-NN.
* `DUMMY_TEST_DIR`: Directory where the dummy test set will be created/looked for.
* `DUMMY_SUBMISSION_FILE`: Path to save the prediction CSV file when using the dummy test set.
* `TEST_DIR`: Path to the *actual* test images (used if `use_dummy_test_set` in `main()` is `False`).
* `NORMALIZE_FEATURES`: `True` to normalize features (e.g., L2 norm) before k-NN training/prediction. Often beneficial.
* `KNN_NEIGHBORS`: Number of neighbors (`k`) for the k-NN classifier.
* `KNN_METRIC`: Distance metric for k-NN ('cosine' or 'euclidean').
* `KNN_WEIGHTS`: Weight function for k-NN ('uniform' or 'distance'). 'distance' often performs better.
* `MODEL_NAME`: Specifies the base CNN architecture (e.g., 'resnet50'). Currently hardcoded to ResNet50 in `load_model`, but designed for extension.
* `DUMMY_SAMPLE_SUBMISSION_FILE`: Path to the ground truth file for the dummy test set, used for evaluation.
* `FINE_TUNED_MODEL_SAVE_PATH`: Path to save/load the fine-tuned model weights.
* `BATCH_SIZE`: Batch size for processing data through the model.
* `NUM_WORKERS`: Number of parallel workers for loading data.

## Usage

1.  **Configure:** Adjust the global variables in `main.py` as needed (see Configuration section). Pay close attention to `PERFORM_FINE_TUNING` and `USE_FINE_TUNED_FOR_CLASSIFICATION` to select the desired workflow.
2.  **Set Test Set:** Decide whether to use the dummy test set for local testing by setting `use_dummy_test_set = True` or `False` within the `main()` function.
3.  **Run the pipeline:**
    ```bash
    python src/main.py
    ```

### Workflows:

* **Workflow 1: Pre-trained Features + k-NN**
    * Set `PERFORM_FINE_TUNING = False`.
    * The script will load the pre-trained ResNet50 (or other specified model), remove its head, extract features from the training set, train/load a k-NN model, extract features from the test set, and predict using k-NN.
* **Workflow 2: Fine-tuning + k-NN**
    * Set `PERFORM_FINE_TUNING = True`.
    * Set `USE_FINE_TUNED_FOR_CLASSIFICATION = False`.
    * The script will fine-tune the ResNet50 model, save the best weights, load these weights back *without* the classification head, then proceed with k-NN training/prediction as in Workflow 1, but using features from the *fine-tuned* model.
* **Workflow 3: Fine-tuning + Direct Classification**
    * Set `PERFORM_FINE_TUNING = True`.
    * Set `USE_FINE_TUNED_FOR_CLASSIFICATION = True`.
    * The script will fine-tune the ResNet50 model (including its classification head), save the best weights, load the model back *with* its classification head, and then directly predict labels for the test set using this end-to-end fine-tuned model. k-NN steps are bypassed.

### Dummy Test Set Usage:

* If `use_dummy_test_set = True` in `main()`:
    * The script will check for `DUMMY_TEST_DIR`. If it doesn't exist or is empty, it will create it by copying a sample of images from the validation set.
    * It will also create a corresponding ground truth CSV (`DUMMY_SAMPLE_SUBMISSION_FILE`).
    * Predictions will be run on this dummy set and saved to `DUMMY_SUBMISSION_FILE`.
    * Finally, it will attempt to evaluate the accuracy of the predictions against the generated ground truth.

## Outputs

* **Fine-tuned Model:** Saved to `FINE_TUNED_MODEL_SAVE_PATH` (e.g., `models/cat_recognizer_finetuned.pth`) if fine-tuning is performed.
* **k-NN Model:** Saved to `KNN_MODEL_SAVE_PATH` (e.g., `models/cat_recognizer_knn.joblib`) if the k-NN workflow is active and a model isn't loaded.
* **Submission File:** Predictions are saved in CSV format (e.g., `dummy_submission.csv` or potentially another configured path if using the actual test set). The format is typically `id,label`, where `id` is the image filename (without extension) and `label` is the predicted class name.
* **Console Output:** Progress information, timings, evaluation results (if applicable).