from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_dir="data/raw", img_size=224, batch_size=16, return_test=False):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_tf)
    val_data = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_tf)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # -------- OPTIONAL TEST LOADER --------
    if return_test:
        test_data = datasets.ImageFolder(root=f"{data_dir}/test", transform=val_tf)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, train_data.classes

    return train_loader, val_loader, train_data.classes
