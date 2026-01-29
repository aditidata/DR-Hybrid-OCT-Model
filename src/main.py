import torch
import cv2
import numpy as np

from src.dataset.oct_dataset import get_dataloaders
from src.models.cnn_baseline import CNNBaseline
from src.training.train_baseline import train_model
from src.explainability.grad_cam import GradCAM


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, val_loader, classes = get_dataloaders(
        data_dir="data/raw",
        img_size=224,
        batch_size=16
    )

    print("Classes:", classes)

    model = CNNBaseline(num_classes=len(classes))

    # -------- TRAIN MODEL --------
    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=5,
        device=device
    )

    # ================= GRAD-CAM =================
    print("\nRunning Grad-CAM visualization...")

    model.eval()

    # Take one validation image
    sample_imgs, sample_labels = next(iter(val_loader))
    sample_img = sample_imgs[0].unsqueeze(0).to(device)

    # Target layer for ResNet-18
    target_layer = model.model.layer4[-1].conv2

    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate(sample_img)

    # Convert image tensor to numpy
    img = sample_img.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    overlay = 0.6 * heatmap / 255 + img

    cv2.imwrite(
        "outputs/gradcam/gradcam_example.png",
        np.uint8(255 * overlay)
    )

    print("✅ Grad-CAM saved to outputs/gradcam/gradcam_example.png")


if __name__ == "__main__":
    main()
