import torch
from sklearn.metrics import confusion_matrix, classification_report

from src.dataset.oct_dataset import get_dataloaders
from src.models.cnn_baseline import CNNBaseline

# ---------------- Device ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------------- Load Data ----------------
train_loader, val_loader, test_loader, classes = get_dataloaders(
    data_dir="data/raw",
    img_size=224,
    batch_size=16,
    return_test=True
)

print("Test Classes:", classes)

# ---------------- Load Model ----------------
model = CNNBaseline(num_classes=len(classes))
model.load_state_dict(
    torch.load("outputs/checkpoints/model.pth", map_location=device)
)
model.to(device)
model.eval()
torch.set_grad_enabled(False)

# ---------------- Evaluation ----------------
all_preds = []
all_labels = []

for imgs, labels in test_loader:
    imgs = imgs.to(device)

    outputs = model(imgs)
    preds = outputs.argmax(dim=1).cpu().numpy()

    all_preds.extend(preds)
    all_labels.extend(labels.numpy())

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
