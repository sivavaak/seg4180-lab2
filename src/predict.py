import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import MODEL_CHECKPOINT_PATH, IMAGE_SIZE, OUTPUT_DIR
from model import create_unet


def load_model(device):
    model = create_unet().to(device)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def predict_single(model, image_input, device):
    if isinstance(image_input, str):
        image = np.array(Image.open(image_input).convert("RGB"))
    else:
        image = np.array(image_input.convert("RGB"))

    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    input_tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = (torch.sigmoid(output) > 0.5).float()

    pred_mask = pred.squeeze().cpu().numpy()
    return image, pred_mask


def visualize(image_path, gt_mask_path=None, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    image, pred_mask = predict_single(model, image_path, device)

    n_cols = 3 if gt_mask_path else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    if gt_mask_path:
        gt_mask = np.array(Image.open(gt_mask_path).convert("L"))
        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")
    else:
        axes[1].imshow(pred_mask, cmap="gray")
        axes[1].set_title("Prediction")
        axes[1].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "prediction.png"))
    args = parser.parse_args()

    visualize(args.image, args.mask, args.output)
