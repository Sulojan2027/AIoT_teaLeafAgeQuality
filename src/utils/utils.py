import torch

def get_predictions(model, loader, device):
    """
    Get predictions from the model on a given loader.
    
    Args:
        model: PyTorch model
        loader: DataLoader
        device: Device to run on
        
    Returns:
        y_true: List of true labels
        y_pred: List of predicted labels
    """
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return y_true, y_pred