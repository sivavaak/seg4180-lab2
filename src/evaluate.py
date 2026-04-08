import os
import torch
from torch.utils.data import DataLoader

from config import IMAGE_SIZE, MODEL_CHECKPOINT_PATH, BATCH_SIZE, NUM_WORKERS
from dataset import AerialSegmentationDataset, get_val_transforms, prepare_dataset, PROCESSED_DIR
from model import create_unet


def compute_iou(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def compute_dice(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return ((2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)).item()


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prepare_dataset()

    dataset = AerialSegmentationDataset(
        os.path.join(PROCESSED_DIR, "test", "images"),
        os.path.join(PROCESSED_DIR, "test", "masks"),
        transform=get_val_transforms(IMAGE_SIZE),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = create_unet().to(device)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_iou += compute_iou(preds, masks)
            total_dice += compute_dice(preds, masks)
            n_batches += 1

    mean_iou = total_iou / n_batches
    mean_dice = total_dice / n_batches
    print(f"Mean IoU:  {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    return mean_iou, mean_dice


if __name__ == "__main__":
    evaluate()
