import torch
from PIL import Image


def predict_image(model, image_path, transform, class_names, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    index = int(probabilities.argmax())
    return {
        "class_id": index,
        "class_name": class_names[index],
        "confidence": float(probabilities[index]),
    }
