import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset.hybrid_dataset import HybridOCTDataset
from src.models.hybrid_model import HybridCNNMF


def train_hybrid(
    data_dir="data/raw",
    img_size=224,
    batch_size=8,
    epochs=5,
    lr=1e-4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------------- Datasets ----------------
    train_data = HybridOCTDataset(
        data_dir=data_dir,
        split="train",
        img_size=img_size
    )

    val_data = HybridOCTDataset(
        data_dir=data_dir,
        split="val",
        img_size=img_size
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False
    )

    # ---------------- Model ----------------
    model = HybridCNNMF(
        mf_dim=3,
        num_classes=3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ---------------- Training ----------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, mf_feats, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}"
        ):
            imgs = imgs.to(device)
            mf_feats = mf_feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, mf_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---------------- Validation ----------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, mf_feats, labels in val_loader:
                imgs = imgs.to(device)
                mf_feats = mf_feats.to(device)
                labels = labels.to(device)

                outputs = model(imgs, mf_feats)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={avg_loss:.4f}, "
            f"Val Acc={val_acc:.4f}"
        )

    # ---------------- Save model ----------------
    torch.save(
        model.state_dict(),
        "outputs/checkpoints/hybrid_model.pth"
    )
    print("✅ Hybrid model saved to outputs/checkpoints/hybrid_model.pth")
