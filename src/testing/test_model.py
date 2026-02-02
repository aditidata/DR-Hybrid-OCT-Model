import torch
from src.models.cnn_baseline import CNNBaseline

# ---------------- Device ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------------- Load model architecture ----------------
model = CNNBaseline(num_classes=3)

# ---------------- Load trained weights ----------------
model.load_state_dict(
    torch.load("outputs/checkpoints/model.pth", map_location=device)
)

# ---------------- Move to device ----------------
model = model.to(device)

# ---------------- Evaluation mode ----------------
model.eval()
torch.set_grad_enabled(False)

print("✅ Model loaded successfully")
print("✅ Model set to evaluation mode")
print("✅ Gradients disabled")
