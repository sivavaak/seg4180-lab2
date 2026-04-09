import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    DATA_DIR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    IMAGE_SIZE, MODEL_CHECKPOINT_PATH, NUM_WORKERS, OUTPUT_DIR,
)
from dataset import (
    AerialSegmentationDataset, get_train_transforms, get_val_transforms,
    prepare_dataset, PROCESSED_DIR,
)
from model import create_unet
from evaluate import compute_iou, compute_dice


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def combined_loss(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dl = dice_loss(pred, target)
    return bce + dl


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    prepare_dataset()

    train_dataset = AerialSegmentationDataset(
        os.path.join(PROCESSED_DIR, "train", "images"),
        os.path.join(PROCESSED_DIR, "train", "masks"),
        transform=get_train_transforms(IMAGE_SIZE),
    )
    val_dataset = AerialSegmentationDataset(
        os.path.join(PROCESSED_DIR, "val", "images"),
        os.path.join(PROCESSED_DIR, "val", "masks"),
        transform=get_val_transforms(IMAGE_SIZE),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = create_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    os.makedirs(os.path.dirname(MODEL_CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_iou += compute_iou(preds, masks)
                val_dice += compute_dice(preds, masks)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_iou": round(val_iou, 4),
            "val_dice": round(val_dice, 4),
        })

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val IoU: {val_iou:.4f} - "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    metrics_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    train()
