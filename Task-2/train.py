import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Enhanced training function with proper metrics
def perform_training(model, train_loader, val_loader, criterion, optimizer, num_epochs, DEVICE='cuda'):
    best_f1 = 0
    best_threshold = 0.5
    metrics = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_outputs, train_labels = [], []
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.float().to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store for metrics
            train_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Calculate training metrics
        train_preds = (np.array(train_outputs) > best_threshold).astype(int)
        metrics['train_loss'].append(loss.item())
        metrics['train_f1'].append(f1_score(train_labels, train_preds))
        metrics['train_precision'].append(precision_score(train_labels, train_preds))
        metrics['train_recall'].append(recall_score(train_labels, train_preds))

        # Validation phase
        model.eval()
        val_outputs_list, val_labels_list = [], []  # Store tensors directly
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.float().to(DEVICE)
                outputs = model(images)
                val_outputs_list.append(outputs)  # Keep raw logits on DEVICE
                val_labels_list.append(labels)    # Keep labels on DEVICE

        # Concatenate tensors after the loop
        all_val_outputs = torch.cat(val_outputs_list, dim=0)
        all_val_labels = torch.cat(val_labels_list, dim=0)

        # Calculate validation loss on DEVICE
        val_loss = criterion(all_val_outputs, all_val_labels).item() # Use 1D labels consistent with 1D outputs
        metrics['val_loss'].append(val_loss)

        # Move to CPU/NumPy for sklearn metrics
        val_outputs_np = torch.sigmoid(all_val_outputs).cpu().numpy()
        val_labels_np = all_val_labels.cpu().numpy()

        # Find optimal threshold using NumPy arrays
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [f1_score(val_labels_np, (val_outputs_np > t).astype(int)) for t in thresholds]
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        
        # Calculate other validation metrics using NumPy arrays
        val_preds_np = (val_outputs_np > best_threshold).astype(int)
        metrics['val_f1'].append(f1_scores[best_idx])
        metrics['val_precision'].append(precision_score(val_labels_np, val_preds_np))
        metrics['val_recall'].append(recall_score(val_labels_np, val_preds_np))

        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train F1: {metrics["train_f1"][-1]:.3f} | '
              f'Val F1: {metrics["val_f1"][-1]:.3f} | '
              f'Threshold: {best_threshold:.3f}')

        # Save best model
        if metrics['val_f1'][-1] > best_f1:
            best_f1 = metrics['val_f1'][-1]
            torch.save(model.state_dict(), 'best_model.pth')

    # Plotting code would go here...
    return model, best_threshold