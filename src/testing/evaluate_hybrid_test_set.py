import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from src.dataset.hybrid_dataset import HybridOCTDataset
from src.models.hybrid_model import HybridCNNMF


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------------- Load Test Dataset ----------------
    test_dataset = HybridOCTDataset(
        data_dir="data/raw",
        split="test",
        img_size=224
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False
    )

    # ---------------- Load Hybrid Model ----------------
    model = HybridCNNMF(
        mf_dim=3,
        num_classes=3
    ).to(device)

    model.load_state_dict(
        torch.load("outputs/checkpoints/hybrid_model.pth", map_location=device)
    )

    model.eval()
    print("✅ Hybrid model loaded")

    # ---------------- Evaluation ----------------
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, mf_feats, labels in test_loader:
            imgs = imgs.to(device)
            mf_feats = mf_feats.to(device)
            labels = labels.to(device)

            outputs = model(imgs, mf_feats)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ---------------- Metrics ----------------
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=["Normal (0)", "DR (1)", "Confounding (3)"]
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
