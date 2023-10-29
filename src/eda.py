import os
import fiftyone as fo
import fiftyone.brain as fob
from typing import Any, Dict, List, Optional, Tuple
import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
import torch
from torchvision.datasets import ImageFolder
import timm
from torchsummary import summary
from omegaconf import DictConfig
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def fiftyone_vis(dataset_dir: str):
    # Create the dataset
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.ImageClassificationDirectoryTree,
    )
    fob.compute_uniqueness(dataset)
    session = fo.launch_app(dataset)
    rank_view = dataset.sort_by("uniqueness")
    session.view = rank_view
    session.wait()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def view_model_transforms(cfg: DictConfig):
    dataset_path = "data/Documents"
    dataset = ImageFolder(dataset_path)

    model = hydra.utils.instantiate(cfg.model, class_to_idx=dataset.class_to_idx)
    # model = timm.create_model("", pretrained=True, num_classes=4)
    data_config = timm.data.resolve_model_data_config(model.net)
    train_transforms = timm.data.create_transform(**data_config, is_training=True)
    print(train_transforms)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def view_model(cfg: DictConfig):
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
    # # summary(model, (3, 504, 504))
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.setup(stage='fit')
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(trainable)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def visualize(cfg: DictConfig):
    original_dataset = 'data/Documents'
    train_dataset = 'data/train'
    val_dataset = 'data/val'

    fiftyone_vis(val_dataset)


if __name__ == '__main__':
    # visualize()
    # view_model_transforms()
    view_model()
