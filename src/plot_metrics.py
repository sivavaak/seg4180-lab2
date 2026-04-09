import os
import json
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


def plot_metrics(history_path=None, save_dir=None):
    if history_path is None:
        history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    if save_dir is None:
        save_dir = OUTPUT_DIR

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_iou = [h["val_iou"] for h in history]
    val_dice = [h["val_dice"] for h in history]

    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_iou, label="Val IoU", linewidth=2)
    ax2.plot(epochs, val_dice, label="Val Dice", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved training curves to {path}")
    plt.close()


if __name__ == "__main__":
    plot_metrics()
