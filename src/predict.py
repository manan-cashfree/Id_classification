from pathlib import Path
from PIL import Image
from typing import BinaryIO, Tuple
import hydra
from hydra import initialize, compose
from lightning import LightningModule
import torch
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.document_module import DocumentLitModule
from src.data.components import val_transforms


def get_model() -> LightningModule:
    """fetch the trained model in eval mode for inference"""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="prod")
        torch.hub.set_dir(cfg.paths.torch_hub_dir)
        net = hydra.utils.instantiate(cfg.model.net)
        doc_model: LightningModule = DocumentLitModule.load_from_checkpoint(cfg.ckpt_path, net=net,
                                                                            class_to_idx=cfg.model.class_to_idx)
        doc_model.eval()
        return doc_model


def predict(model: LightningModule, img: str | bytes | Path | BinaryIO) -> Tuple[str, float]:
    """

    Args:
        model (LightningModule): trained model
        img: input image used for prediction
    """
    # load image and apply transforms
    pil_img = Image.open(img)
    pil_img = val_transforms(pil_img)
    if pil_img.dim() == 3:
        pil_img = pil_img.unsqueeze(0)
    with torch.no_grad():
        pred, conf = model.inference_step(pil_img)
        return pred, conf


if __name__ == '__main__':
    document_model = get_model()
    predict(document_model, "src/data/test.jpg")
