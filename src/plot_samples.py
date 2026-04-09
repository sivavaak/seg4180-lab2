import os
import random
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from config import IMAGE_SIZE, OUTPUT_DIR
from dataset import prepare_dataset, PROCESSED_DIR
from predict import load_model, predict_single


def plot_samples(n_samples=4, split="test", seed=42, save_path=None):
    prepare_dataset()

    images_dir = os.path.join(PROCESSED_DIR, split, "images")
    masks_dir = os.path.join(PROCESSED_DIR, split, "masks")
    files = sorted(os.listdir(images_dir))

    random.seed(seed)
    selected = random.sample(files, min(n_samples, len(files)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i, fname in enumerate(selected):
        img_path = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir, fname)

        image, pred_mask = predict_single(model, img_path, device)
        gt_mask = np.array(Image.open(mask_path).convert("L"))

        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Aerial Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_mask, cmap="gray")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "sample_predictions.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved sample predictions to {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    plot_samples(args.n, args.split, args.seed, args.output)
