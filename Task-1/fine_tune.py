import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

def fine_tune_model(model, num_input_features, train_loader, val_loader, num_classes, model_dir):
    """
    Performs fine-tuning of the model on the cat dataset.
    (This is a basic structure - requires significant expansion for robust training)
    """
    global DEVICE, NUM_EPOCHS, LEARNING_RATE, OPTIMIZER_TYPE, LOSS_FUNCTION, FINE_TUNED_MODEL_SAVE_PATH

    print(f"\n--- Starting Fine-Tuning ---")

    # --- 1. Add Classification Head ---
    # Replace the nn.Identity() added during initial loading with a proper classifier
    print(f"Adding classification head for {num_classes} classes.")
    classifier_head = nn.Linear(num_input_features, num_classes)
    model.fc = classifier_head # Replace the head with a new one
    model = model.to(DEVICE) # Ensure model is on the correct device

    # --- 2. Define Loss Function ---
    if LOSS_FUNCTION == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    # Add other loss functions if needed (e.g., Label Smoothing)
    else:
        raise ValueError(f"Unsupported loss function: {LOSS_FUNCTION}")
    print(f"Using loss function: {LOSS_FUNCTION}")

    # --- 3. Define Optimizer ---
    # Fine-tuning often benefits from lower learning rates and optimizing only the head
    # or unfreezing later layers gradually. Here, we optimize all parameters for simplicity.
    # TODO: Implement strategies like freezing backbone layers initially.
    if OPTIMIZER_TYPE == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER_TYPE == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4) # Add momentum/decay for SGD
    else:
        raise ValueError(f"Unsupported optimizer type: {OPTIMIZER_TYPE}")
    print(f"Using optimizer: {OPTIMIZER_TYPE} with LR: {LEARNING_RATE}")

    # --- Learning Rate Scheduler (Optional but Recommended) ---
    # Example: StepLR or ReduceLROnPlateau
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # Reduce LR every 5 epochs
    print("Using StepLR scheduler.")

    # --- Mixed Precision Scaler (Optional but Recommended for Speed/Memory on GPU) ---
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
    print(f"Gradient Scaling enabled: {scaler.is_enabled()}")


    # --- 4. Training Loop ---
    print(f"Training for {NUM_EPOCHS} epochs...")
    best_val_accuracy = 0.0
    best_train_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        _continue =  input("Continue training? (y/n): ").strip().lower()
        if _continue != 'y':
            print("Training interrupted by user.")
            return model
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_start_time = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            # Mixed precision context
            with autocast(enabled=scaler.is_enabled()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Print progress every batche
            print(f' Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}', end='\r')


        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc_train = correct_train / total_train
        train_end_time = time.time()
        print(f'\nEpoch {epoch+1} Training Finished. Loss: {epoch_loss:.4f}, Acc: {epoch_acc_train:.4f}, Time: {train_end_time - train_start_time:.2f}s')

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0
        val_start_time = time.time()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                with autocast(enabled=scaler.is_enabled()): # Use autocast for validation too
                     outputs = model(inputs)
                     loss = criterion(outputs, labels)

                running_loss_val += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_loss_val = running_loss_val / len(val_loader.dataset)
        epoch_acc_val = correct_val / total_val
        val_end_time = time.time()
        print(f'Epoch {epoch+1} Validation. Loss: {epoch_loss_val:.4f}, Acc: {epoch_acc_val:.4f}, Time: {val_end_time - val_start_time:.2f}s')

        # Update learning rate
        scheduler.step()
        print(f" Current LR: {optimizer.param_groups[0]['lr']}")


        # --- Save Best Model ---
        if epoch_acc_train > best_train_accuracy:
            print(f"Training accuracy improved ({best_train_accuracy:.4f} -> {epoch_acc_train:.4f}). Saving model...")
            best_train_accuracy = epoch_acc_train
            try:
                # Save the model state_dict
                os.makedirs(os.path.dirname(FINE_TUNED_MODEL_SAVE_PATH), exist_ok=True)
                # If using DataParallel, save module's state_dict
                if isinstance(model, nn.DataParallel):
                     torch.save(model.module.state_dict(), FINE_TUNED_MODEL_SAVE_PATH)
                else:
                     torch.save(model.state_dict(), FINE_TUNED_MODEL_SAVE_PATH)
                print(f"Model saved to {FINE_TUNED_MODEL_SAVE_PATH}")
            except Exception as e:
                print(f"Error saving model: {e}")

        try:
            os.makedirs(os.path.dirname(FINE_TUNED_MODEL_SAVE_PATH), exist_ok=True)
            if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), os.path.join(model_dir, f'cat_recognizer_finetuned_{epoch}.pth'))
            else:
                    torch.save(model.state_dict(), os.path.join(model_dir, f'cat_recognizer_finetuned_{epoch}.pth'))
            print(f"Model saved to {os.path.join(model_dir, f'cat_recognizer_finetuned_{epoch}.pth')}")
        except Exception as e:
            print(f"Error saving model: {e}")

    print(f"--- Fine-Tuning Finished ---")
    print(f"Best Validation Accuracy achieved: {epoch_acc_train:.4f}")

    # Return the *best* model state (by reloading the saved one)
    # Or just return the model as it is after the last epoch
    # For simplicity, return the model in its current state
    return model