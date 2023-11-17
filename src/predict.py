import os
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule
import torch
from torchvision.datasets import ImageFolder
from omegaconf import DictConfig
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.document_module import DocumentLitModule
from src.data.components import val_transforms

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def predict(cfg: DictConfig, img_name:str = 'test.jpg'):
    net = hydra.utils.instantiate(cfg.model.net)
    model = DocumentLitModule.load_from_checkpoint(cfg.ckpt_path, net=net, class_to_idx=cfg.model.class_to_idx)
    model.eval()
    # load image and apply transforms
    img = Image.open(os.path.join(cfg.paths.data_dir, img_name))
    img = val_transforms(img)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred, conf = model.inference_step(img)
        print(pred, conf)


if __name__ == '__main__':
    predict()