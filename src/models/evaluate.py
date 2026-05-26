import torch
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(model, dataloader, device, target_names=None):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.tolist())

    return {
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
