"""
Train a custom hazard classifier (flood / fire / landslide / earthquake / normal)
------------------------------------------------------------------------------------
Transfer learning on ResNet18 using your own labeled images.

Dataset folder structure required (ImageFolder format):

    data/
      train/
        flood/            *.jpg
        fire/             *.jpg
        landslide_debris/ *.jpg
        earthquake_damage/*.jpg
        normal/           *.jpg
      val/
        flood/ ...
        fire/  ...
        ...

Collect images from: drone footage frames, public disaster-image datasets
(e.g. Kaggle "Disaster Images Dataset", xBD building-damage dataset for
earthquake/landslide debris), or hand-labeled frames from CAM01.mp4.

Usage:
    python train_hazard_classifier.py --data ./data --epochs 10 --out hazard_best.pt
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_dataloaders(data_dir, batch_size=16):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"{data_dir}/val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_ds.classes


def train(data_dir, epochs, out_path, lr=1e-4, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(data_dir, batch_size)
    print(f"[train] Classes found: {class_names}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else 0.0

        print(f"[train] Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": class_names}, out_path)
            print(f"[train] Saved new best model to {out_path} (val_acc={val_acc:.4f})")

    print(f"[train] Done. Best val_acc: {best_acc:.4f}. Model saved at: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data", help="Path to dataset root (with train/ and val/)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="hazard_best.pt")
    args = parser.parse_args()

    train(args.data, args.epochs, args.out, args.lr, args.batch_size)
