import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.features.multifractal import extract_multifractal_features

LABEL_MAP = {
    "0": 0,
    "1": 1,
    "3": 2
}

class HybridOCTDataset(Dataset):
    def __init__(self, data_dir, split="train", img_size=224):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        split_dir = os.path.join(data_dir, split)

        for folder_name, mapped_label in LABEL_MAP.items():
            class_dir = os.path.join(split_dir, folder_name)
            for fname in os.listdir(class_dir):
                self.samples.append(
                    (os.path.join(class_dir, fname), mapped_label)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img_pil)

        img_np = np.array(img_pil)
        mf_features = extract_multifractal_features(img_np)
        mf_features = torch.tensor(mf_features, dtype=torch.float32)

        return img_tensor, mf_features, label
