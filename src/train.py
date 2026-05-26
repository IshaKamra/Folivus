import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import PlantDataset, get_transforms
from models import FolivusNet
from utils import log_folivus_banner, save_metrics_plot


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(dataloader, desc="  Training Batches", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def main():
    log_folivus_banner()

    # 1. Load configuration yaml file
    config_path = os.path.join("config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["training"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    elif device.type == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    print(f"[INFO] Initialized optimization runtime environment on: {device}\n")

    # 2. Dataset Partition Split Pipeline
    print("[INFO] Assembling data split sets...")
    full_dataset = PlantDataset(root_dir=config["data"]["raw_dir"])

    total_size = len(full_dataset)
    train_size = int(config["data"]["train_split"] * total_size)
    val_size = int(config["data"]["val_split"] * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Set distinct augmentation configurations across subsets
    train_set.dataset.transform = get_transforms(
        config["data"]["img_size"], augment=True
    )
    val_set.dataset.transform = get_transforms(
        config["data"]["img_size"], augment=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    # 3. Network Compilation
    model = FolivusNet(
        num_classes=config["model"]["num_classes"],
        dropout_rate=config["model"]["dropout_rate"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    # 4. Main Supervised Optimization Loop
    epochs = config["training"]["epochs"]
    print(f"[INFO] Initiating training workflow across {epochs} target epochs...")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ➔ "
            f"Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(
                os.path.dirname(config["training"]["model_save_path"]), exist_ok=True
            )
            torch.save(model.state_dict(), config["training"]["model_save_path"])
            print(f"  🏆 Objective target beaten. Saved optimal weight map.")

    print("\n[INFO] Saving analytics visualization reports...")
    save_metrics_plot(history, output_path="reports/metrics_plot.png")
    print("[SUCCESS] Folivus AI training engine execution complete.")


if __name__ == "__main__":
    main()