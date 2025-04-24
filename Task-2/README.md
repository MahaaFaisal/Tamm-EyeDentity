# PyTorch Binary Image Classification Project

This project implements a binary image classification pipeline using PyTorch. It trains a Convolutional Neural Network (ConvNet) on a dataset, handles class imbalance, performs validation to find the best model and optimal prediction threshold, and finally generates predictions on a test set, saving the results to a submission file.

## Features

* **Convolutional Neural Network (ConvNet):** Uses a custom CNN architecture (defined in `models.py`).
* **PyTorch Framework:** Built entirely using PyTorch for model definition, training, and inference.
* **Data Handling:** Uses PyTorch `Dataset` and `DataLoader` for efficient data loading and batching. Custom functions load data lists from CSV files.
* **Image Transformations:** Applies separate transformations for training, validation, and testing data (defined in `transformers.py`).
* **Class Imbalance Handling:** Uses `pos_weight` in `BCEWithLogitsLoss` to give more importance to the positive class, suitable for imbalanced datasets.
* **Training & Validation:** Includes a training loop (`perform_training` function in `train.py`) that evaluates the model on a validation set during training.
* **Best Model Saving:** Saves the state dictionary of the model with the best validation performance.
* **Optimal Threshold:** Determines an optimal probability threshold for classification based on validation performance (returned by `perform_training`).
* **Inference:** Loads the best model to make predictions on the test set.
* **Submission File Generation:** Creates a `submission.csv` file in the specified format.
* **GPU Acceleration:** Automatically utilizes a CUDA-enabled GPU if available, otherwise falls back to CPU.
* **Reproducibility:** Sets a fixed random seed for PyTorch operations.

## Prerequisites

* Python 3.x
* PyTorch (`torch`)
* Pillow (`PIL`)
* (Assumed) NumPy (`numpy`) - Often used with PyTorch and data loading.

You can typically install the main dependencies using pip:
```bash
pip install torch pillow numpy