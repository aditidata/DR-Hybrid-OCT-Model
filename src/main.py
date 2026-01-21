import torch
from src.dataset.oct_dataset import get_dataloaders
from src.models.cnn_baseline import CNNBaseline
from src.training.train_baseline import train_model


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
    train_model(model, train_loader, val_loader, epochs=5, device=device)


if __name__ == "__main__":
    main()
