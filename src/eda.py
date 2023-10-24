import os
import fiftyone as fo
from typing import Any, Dict, List, Optional, Tuple
import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
import torch
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
    session = fo.launch_app(dataset)
    session.wait()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    originl_dataset = 'data/Documents'
    train_dataset = 'data/train'
    val_dataset = 'data/val'

    model = hydra.utils.instantiate(cfg.model)
    # for name, param in model.named_parameters():
    #     if 'head' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    summary(model.net, (3, 518, 518))
    print(f'trainable_params: {trainable_params}')

    # data_config = timm.data.resolve_model_data_config(model)
    # print(type(data_config))
    # train_transforms = timm.data.create_transform(**data_config, is_training=True)
    # val_transforms = timm.data.create_transform(**data_config, is_training=False)
    # print(train_transforms)
    # print(val_transforms)

    # print(f"Instantiating datamodule <{cfg.data._target_}>")
    # datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, data_config=data_config)
    # datamodule.setup()
    # fiftyone_vis(val_dataset)


if __name__ == '__main__':
    main()
