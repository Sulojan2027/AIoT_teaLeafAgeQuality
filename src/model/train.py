from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, accuracy_score

def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # tqdm adds a nice progress bar in your terminal
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # 1. Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 2. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. Track metrics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    # Step the scheduler after each epoch (if provided)
    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    # Use 'macro' average to treat all 4 tea age classes equally
    epoch_f1 = f1_score(all_labels, all_preds, average='macro') 

    return epoch_loss, epoch_acc, epoch_f1

def evaluate_epoch(model, loader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad(): # No gradients needed for validation (saves memory/time)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, epoch_acc, epoch_f1