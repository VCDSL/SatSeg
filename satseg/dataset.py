import os
import random
from glob import glob
from time import time
import numpy as np
import torchvision.transforms.functional as TF
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader

from satseg.geo_tools import generate_mask, tif2np


class CustomDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, indices: List[int]):
        super().__init__()
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.npy")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.npy")))
        self.indices = indices

        self.n_channels = float("inf")
        for path in self.image_paths:
            self.n_channels = min(self.n_channels, np.load(path).shape[0])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        image = np.load(self.image_paths[index])[: self.n_channels].transpose(1, 2, 0)
        mask = np.load(self.mask_paths[index])

        return TF.to_tensor(image).float(), TF.to_tensor(mask).float()


class InferenceDataset(Dataset):
    def __init__(self, image_dir: str):
        super().__init__()
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.npy")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = np.load(path).transpose(1, 2, 0)

        return TF.to_tensor(image).float(), path


def create_datasets(
    tif_paths: List[str],
    mask_paths: List[str],
    data_dir: str,
    image_size: int = 256,
    stride: int = None,
    train_pct: int = None,
) -> Tuple[CustomDataset, CustomDataset]:
    print("Creating dataset...")
    assert len(tif_paths) == len(mask_paths)

    if not stride:
        stride = image_size
    if not train_pct:
        train_pct = 80

    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    tot_count = 0
    for tif_path, mask_path in zip(tif_paths, mask_paths):
        count = np2images(tif_path, mask_path, image_size, stride, data_dir)
        tot_count += count

    print(f"Total count: {tot_count}")

    indices = list(range(tot_count))
    random.shuffle(indices)

    train_size = int(tot_count * train_pct / 100)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_set = CustomDataset(image_dir, mask_dir, train_indices)
    val_set = CustomDataset(image_dir, mask_dir, val_indices)
    print(
        f"Dataset created! Train set size: {len(train_set)}, Val set size: {len(val_set)}"
    )

    return (train_set, val_set)


def create_inference_dataset(tif_paths: List[str], data_dir: str, image_size: int):
    image_dir = os.path.join(data_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    for tif_path in tif_paths:
        np2images(tif_path, None, image_size, image_size, data_dir)

    return InferenceDataset(image_dir)


def np2images(
    tif_path: str,
    mask_path: Optional[str],
    image_size: int,
    stride: int,
    save_dir: str = "",
) -> int:
    tif_name = os.path.basename(tif_path)[: -len(".tif")]
    image_arr = tif2np(tif_path)
    image_dir = os.path.join(save_dir, "images")

    _, h, w = image_arr.shape
    print(f"Image shape: {image_arr.shape}")

    count = 0
    if mask_path:
        mask_dir = os.path.join(save_dir, "masks")
        mask_arr = generate_mask(tif_path, mask_path)
        for i in np.arange(0, h - image_size, stride):
            for j in np.arange(0, w - image_size, stride):
                mask = mask_arr[i : i + image_size, j : j + image_size]
                if np.sum(mask) > 0:
                    count += 1
                    image = image_arr[:, i : i + image_size, j : j + image_size]

                    np.save(os.path.join(image_dir, f"{tif_name}_{i}_{j}.npy"), image)
                    np.save(os.path.join(mask_dir, f"{tif_name}_{i}_{j}.npy"), mask)
    else:
        for i in np.arange(0, h - image_size, stride):
            for j in np.arange(0, w - image_size, stride):
                count += 1
                image = image_arr[:, i : i + image_size, j : j + image_size]
                np.save(os.path.join(image_dir, f"{tif_name}_{i}_{j}.npy"), image)

    return count


def create_dataloader(dataset: Dataset, is_train: bool):
    return DataLoader(dataset, batch_size=4, shuffle=is_train, num_workers=0)
