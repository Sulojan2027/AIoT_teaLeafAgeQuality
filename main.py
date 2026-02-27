import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# Import your custom modules
from src.data_utils.dataloader import create_dataloaders
from src.model.model import create_model
from src.utils.plot import plot_confusion_matrix, plot_training_curves
from src.utils.utils import get_predictions
from src.model.train import train_epoch, evaluate_epoch

# ==========================================
DATA_DIR = 'data'
MODEL_NAME = 'efficientnet_b0'
NUM_CLASSES = 4
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
STEP_SIZE = 5
GAMMA = 0.1
SAVE_PATH = 'weights/best_model.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ==========================================

def main():
    print("========================================")
    print("Starting Training Pipeline")
    print("========================================\n")
    print(f"Device: {DEVICE.upper()}")
    print(f"Model:  {MODEL_NAME}")
    print(f"Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE}\n")

    # 1. Prepare Data
    train_loader, valid_loader, test_loader = create_dataloaders(
        data_dir=DATA_DIR, 
        batch_size=BATCH_SIZE
    )

    # 2. Initialize Model
    model = create_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)

    # 3. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # 4. Setup Trackers for Plotting
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_f1 = 0.0

    print("\nStarting Training...\n")
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scheduler=scheduler)
        
        # Validate
        val_loss, val_acc, val_f1 = evaluate_epoch(model, valid_loader, criterion, DEVICE)

        # Track metrics for the learning curves
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print Epoch Results
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Valid - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Save the best model based on Validation F1-Score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH)
        
        print("-" * 40)

    print(f"\nTraining Complete! Best Validation F1-Score: {best_val_f1:.4f}")
    print(f"Model weights saved to {SAVE_PATH}")

    # 5. Generate Training Curves
    print("\nGenerating Training Curves...")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, EPOCHS)

    print("\nStarting Evaluation...\n")
    
    # 6. Load the weights of the best model from your disk
    model.load_state_dict(torch.load(SAVE_PATH))

    # Evaluate using the unseen test data to get final metrics
    test_loss, test_acc, test_f1 = evaluate_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=DEVICE
    )

    # 7. Collect raw predictions for the Confusion Matrix
    print("Collecting predictions for Confusion Matrix...")
    y_true, y_pred = get_predictions(model, test_loader, device=DEVICE)

    # Define the class names for the plot labels
    class_names = ['1-2 Days', '3-4 Days', '5-7 Days', '7+ Days']

    # Generate Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, class_names, test_loss, test_acc, test_f1)

    print("========================================")
    print(f"Final Test Results")
    print(f"Test - Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    print("========================================\n")

if __name__ == '__main__':
    main()