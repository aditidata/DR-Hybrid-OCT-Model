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

    # -------- SAVE TRAINED MODEL --------
    torch.save(model.state_dict(), "outputs/checkpoints/model.pth")
    print("✅ Model saved to outputs/checkpoints/model.pth")

    # ================= MULTI IMAGE GRAD-CAM =================
    print("\nRunning Grad-CAM for multiple validation images...")

    model.eval()

    # Target layer for ResNet-18
    target_layer = model.model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer)

    save_count = 0
    max_images = 60   # number of Grad-CAM images to save

    for imgs, labels in val_loader:
        imgs = imgs.to(device)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        for i in range(imgs.size(0)):
            if save_count >= max_images:
                break

            img_tensor = imgs[i].unsqueeze(0)

            # Generate Grad-CAM
            cam = grad_cam.generate(img_tensor)

            # Convert tensor image to numpy
            img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

            # Create heatmap and overlay
            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam),
                cv2.COLORMAP_JET
            )

            overlay = 0.6 * heatmap / 255 + img_np
            overlay = np.uint8(255 * overlay)

            pred_class = preds[i].item()

            filename = f"outputs/gradcam/sample_{save_count}_pred{pred_class}.png"
            cv2.imwrite(filename, overlay)

            print(f"Saved {filename}")
            save_count += 1

        if save_count >= max_images:
            break

    print("✅ Multiple Grad-CAM images saved in outputs/gradcam/")


if __name__ == "__main__":
    main()
