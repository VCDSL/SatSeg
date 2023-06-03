import os
from tqdm import tqdm
from typing import Optional, Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torchgeo.models import resnet50, ResNet50_Weights

from satseg.dataset import create_dataloader
from satseg.utils import device, jaccard_index


def create_new_model(
    arch: str,
    channels: int,
    weights: Optional[ResNet50_Weights] = ResNet50_Weights.SENTINEL2_ALL_MOCO,
) -> nn.Module:
    if arch.lower() == "unet":
        model = smp.Unet(encoder_name="resnet50", in_channels=channels, classes=1)

        if weights:
            model_resnet = resnet50(weights)
            del model_resnet.fc
            del model_resnet.global_pool

            for layer1, layer2 in zip(
                list(model.encoder.children())[1:], list(model_resnet.children())[1:]
            ):
                layer1.load_state_dict(layer2.state_dict())

        model.segmentation_head[2].activation = nn.Sigmoid()

    model.to(device)
    model = model.float()

    return model


def train_model(
    train_set: Dataset, val_set: Dataset, arch: str, epochs: int = 10
) -> Tuple[nn.Module, dict]:
    dataloaders = {
        "train": create_dataloader(train_set, True),
        "val": create_dataloader(val_set, False),
    }

    model = create_new_model(arch, val_set[0][0].shape[0])

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    metrics = {"train": {"loss": [], "iou": []}, "val": {"loss": [], "iou": []}}

    print("Training started...")
    for ep in range(epochs):
        print(f"Epoch {ep}")
        for phase in dataloaders:
            for img, mask in tqdm(
                dataloaders[phase], "Training" if phase == "train" else "Validating"
            ):
                img = img.to(device)
                mask = mask.to(device)

                out = model(img)
                loss = criterion(out, mask)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                iou = jaccard_index(out, mask)
                metrics[phase]["loss"].append(loss.item())
                metrics[phase]["iou"].append(iou.item())

    return model, metrics


def run_inference(dataset: Dataset, model: nn.Module, save_dir: str):
    dataloader = create_dataloader(dataset, is_train=False)

    for imgs, paths in tqdm(dataloader, "Running Inference..."):
        imgs = imgs.to(device)

        out = model(imgs).detach().numpy()
        for i, path in enumerate(paths):
            save_path = os.path.join(save_dir, os.path.basename(path))
            np.save(save_path, out[i])


def load_model(model_path: str) -> nn.Module:
    model = torch.load(model_path)
    model.to(device)

    return model


def save_model(model: nn.Module, model_path: str):
    torch.save(model, model_path)
