import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from src.models.cnn_baseline import CNNBaseline
from src.explainability.grad_cam import GradCAM

# path to your downloaded internet image
image_path = "C:/DR-Hybrid-OCT-Model/data/raw/val/1/dr_val_1015.jpg"


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 3 output classes
model = CNNBaseline(num_classes=3)

# load trained weights (make sure this file exists)
model.load_state_dict(torch.load("outputs/checkpoints/model.pth", map_location=device))
model.to(device)
model.eval()

# preprocessing (same size as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# load image
img_pil = Image.open(image_path).convert("RGB")
img_tensor = transform(img_pil).unsqueeze(0).to(device)

# prediction
with torch.no_grad():
    outputs = model(img_tensor)
    pred_class = outputs.argmax(dim=1).item()

print("Predicted class index:", pred_class)

# Grad-CAM
target_layer = model.model.layer4[-1].conv2
grad_cam = GradCAM(model, target_layer)
cam = grad_cam.generate(img_tensor)

# convert tensor image to numpy for overlay
img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = np.uint8(255 * (0.6 * heatmap / 255 + img_np))

save_path = f"outputs/gradcam/single_test_pred{pred_class}.png"
cv2.imwrite(save_path, overlay)

print("Saved Grad-CAM to:", save_path)
