import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from PIL import Image
from transformers import test_transform, train_transform, val_transform
from models import ConvNet, ImageDataset
from train import perform_training
from load_data import load_data_from_train_file, load_data_from_val_file, load_data_from_test_file

# Set random seeds for reproducibility
torch.manual_seed(42)

# Define hyperparameters
BATCH_SIZE = 64  # Increased batch size for better gradient estimation
LEARNING_RATE = 0.0001  # Reduced learning rate for stability
NUM_EPOCHS = 40
CLASS_WEIGHT = 20.0  # For 1:20 imbalance ratio

# Explicitly use GPU 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def predict(model, image_path, best_threshold=0.5):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(image)
        print('output is ' ,output)
        predicted = 1 if output.item() >= best_threshold else 0
        confidence = output.item() if predicted == 1 else 1 - output.item()
    
    return predicted, confidence


def main():
    # Load data
    train_paths, train_labels = load_data_from_train_file('dataset/train.csv')
    val_paths, val_labels = load_data_from_val_file('dataset/val.csv')



    # Create datasets and loaders
    train_dataset = ImageDataset(train_paths, train_labels, train_transform)
    val_dataset = ImageDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize model with class weighting
    model = ConvNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CLASS_WEIGHT]).to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Train and get optimal threshold
    model, best_threshold = perform_training(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE )

    # Load best model for inference
    torch.save(model.state_dict(), 'best_model.pth')
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    test_paths = load_data_from_test_file('dataset/test.csv')
    paths = []
    labels = []
    for image_path in test_paths:
        # Predict on test images
        predicted, _ = predict(model, image_path)
        paths.append(image_path)
        labels.append(predicted)
    
    # Save results
    print("Saving results to submission.csv")
    with open('submission.csv', 'w') as f:
        f.write('path,label\n')
        for i, path in enumerate(paths):
            f.write(f'{path.replace("dataset/test/", "")},{labels[i]}\n')

if __name__ == "__main__":
    main()
