import torch
import torch.nn as nn
from torchvision import models


class HybridCNNMF(nn.Module):
    """
    Hybrid CNN + Multifractal Feature Model
    """

    def __init__(self, num_classes, mf_dim=10):
        super().__init__()

        # CNN backbone (ResNet18)
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        cnn_out_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()  # remove classifier

        # Multifractal feature branch
        self.mf_branch = nn.Sequential(
            nn.Linear(mf_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, mf_features):
        cnn_features = self.cnn(img)
        mf_out = self.mf_branch(mf_features)

        fused = torch.cat((cnn_features, mf_out), dim=1)
        return self.classifier(fused)
