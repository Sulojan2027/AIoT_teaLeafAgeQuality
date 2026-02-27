import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, epochs):
    """
    Plot training and validation loss/accuracy curves.
    """
    epochs_range = np.arange(1, epochs + 1)
    
    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss Curves
    ax1.plot(epochs_range, train_losses, 'b-', marker='o', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, val_losses, 'r-', marker='s', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs_range[::max(1, len(epochs_range)//10)])
    
    # Plot 2: Accuracy Curves
    ax2.plot(epochs_range, train_accs, 'b-', marker='o', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs_range, val_accs, 'r-', marker='s', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs_range[::max(1, len(epochs_range)//10)])
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved to 'training_curves.png'")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, test_acc, test_f1):
    """
    Plot confusion matrix for test data
    """
    print("Generating Confusion Matrix...")
    
    # 1. Generate the raw matrix numbers
    cm = confusion_matrix(y_true, y_pred)

    # 2. Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 3. Use Seaborn's heatmap (bypasses the scikit-learn bug entirely)
    sns.heatmap(
        cm, 
        annot=True,       # Show the numbers inside the boxes
        fmt='d',          # Format as integers
        cmap='Blues',     # Color scheme
        ax=ax, 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    
    # 4. Add a clean title and labels
    plt.title(f'IoTea Confusion Matrix\nTest Acc: {test_acc:.4f} | F1: {test_f1:.4f}', pad=20)
    plt.ylabel('True Age (Actual)')
    plt.xlabel('Predicted Age (Model Output)')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Saved as 'confusion_matrix.png'")
    
    # Free up memory
    plt.close()