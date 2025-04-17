from PIL import Image
import json
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms

def classify_image(image_path):
    """
    Classifies an image using a pre-trained ResNeXt-50 32x4d model.

    Parameters:
        image_path (str): The file path to the image.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing class names and their corresponding probabilities.
    """

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(256),              # Resize smaller edge to 256
        transforms.CenterCrop(224),          # Crop center 224x224
        transforms.ToTensor(),               # Convert PIL image to Tensor
        transforms.Normalize(                # Normalize using ImageNet means/stds
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load and preprocess the image
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('RGB')
    else:
        img = image_path.convert('RGB')  # Ensure it's in RGB mode
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Load the pre-trained model
    model = models.resnext50_32x4d(pretrained=True)
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor)
        # print(outputs.shape)
        _, predicted = torch.max(outputs, 1)

    # Get top-5 predictions
    # top5_prob, top5_catid = torch.topk(probs, 5)

    # Download ImageNet class labels
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urllib.request.urlopen(url) as f:
        imagenet_classes = [line.strip().decode("utf-8") for line in f]

    # Prepare and return the results
    

    return imagenet_classes[predicted.item()]



# image_path = r'C:\Users\kunde\Downloads\pytorch_model\Umbrella.jpg'
# predictions = classify_image(image_path)
# print(predictions)

