import os
import glob
import random
import zipfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    DATA_DIR, KAGGLE_DATASET, BUILDING_COLOR_RGB,
    PATCH_SIZE, TRAIN_RATIO, VAL_RATIO,
)

RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def download_dataset():
    if os.path.exists(RAW_DIR) and any(os.scandir(RAW_DIR)):
        print(f"Raw data already exists at {RAW_DIR}, skipping download.")
        return
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} via Kaggle API...")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=RAW_DIR, unzip=True)

    print("Download complete.")


def find_image_mask_pairs(raw_dir):
    pairs = []

    image_dirs = sorted(glob.glob(os.path.join(raw_dir, "**", "images"), recursive=True))
    mask_dirs = sorted(glob.glob(os.path.join(raw_dir, "**", "masks"), recursive=True))

    mask_dir_map = {}
    for md in mask_dirs:
        parent = os.path.dirname(md)
        mask_dir_map[parent] = md

    for img_dir in image_dirs:
        parent = os.path.dirname(img_dir)
        if parent not in mask_dir_map:
            continue
        msk_dir = mask_dir_map[parent]

        for img_file in sorted(os.listdir(img_dir)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                continue
            stem = os.path.splitext(img_file)[0]
            mask_file = None
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                candidate = stem + ext
                if os.path.exists(os.path.join(msk_dir, candidate)):
                    mask_file = candidate
                    break
            if mask_file:
                pairs.append((
                    os.path.join(img_dir, img_file),
                    os.path.join(msk_dir, mask_file),
                ))

    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs found under {raw_dir}. "
            "Expected subdirectories containing 'images/' and 'masks/' folders "
            "with matching filenames."
        )

    print(f"Found {len(pairs)} image/mask pairs.")
    return pairs


def rgb_mask_to_binary(mask_rgb, target_color):
    r, g, b = target_color
    match = (
        (mask_rgb[:, :, 0] == r) &
        (mask_rgb[:, :, 1] == g) &
        (mask_rgb[:, :, 2] == b)
    )
    return (match.astype(np.uint8) * 255)


def tile_image(img_array, patch_size):
    h, w = img_array.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = img_array[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    return patches


def prepare_dataset():
    splits_exist = all(
        os.path.isdir(os.path.join(PROCESSED_DIR, s, "images"))
        for s in ("train", "val", "test")
    )
    if splits_exist:
        counts = {
            s: len(os.listdir(os.path.join(PROCESSED_DIR, s, "images")))
            for s in ("train", "val", "test")
        }
        if all(c > 0 for c in counts.values()):
            print(f"Processed data already exists: {counts}. Skipping.")
            return

    download_dataset()
    pairs = find_image_mask_pairs(RAW_DIR)

    all_patches = []
    patch_id = 0

    for img_path, mask_path in pairs:
        img = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

        binary_mask = rgb_mask_to_binary(mask_rgb, BUILDING_COLOR_RGB)
        img_patches = tile_image(img, PATCH_SIZE)
        mask_patches = tile_image(binary_mask, PATCH_SIZE)

        for ip, mp in zip(img_patches, mask_patches):
            all_patches.append((patch_id, ip, mp))
            patch_id += 1

    print(f"Generated {len(all_patches)} patches of size {PATCH_SIZE}x{PATCH_SIZE}.")

    random.seed(42)
    random.shuffle(all_patches)

    n = len(all_patches)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": all_patches[:n_train],
        "val": all_patches[n_train:n_train + n_val],
        "test": all_patches[n_train + n_val:],
    }

    for split_name, patches in splits.items():
        img_dir = os.path.join(PROCESSED_DIR, split_name, "images")
        msk_dir = os.path.join(PROCESSED_DIR, split_name, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        for pid, img_patch, mask_patch in patches:
            fname = f"{pid:05d}.png"
            Image.fromarray(img_patch).save(os.path.join(img_dir, fname))
            Image.fromarray(mask_patch).save(os.path.join(msk_dir, fname))

    print(f"Split sizes - train: {len(splits['train'])}, "
          f"val: {len(splits['val'])}, test: {len(splits['test'])}")


def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class AerialSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        image = np.array(Image.open(
            os.path.join(self.images_dir, fname)
        ).convert("RGB"))
        mask = np.array(Image.open(
            os.path.join(self.masks_dir, fname)
        ).convert("L"))

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


if __name__ == "__main__":
    prepare_dataset()
