import os.path as osp
from typing import Any, Dict, Optional, Tuple
import splitfolders
import timm
import torch.utils.data
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchsampler import ImbalancedDatasetSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from src.utils import RankedLogger, ImageFolderWithPaths
import rootutils

log = RankedLogger(__name__, rank_zero_only=True)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class DocumentsDataModule(LightningDataModule):
    """`LightningDataModule` for the Documents dataset.

    The documents include aadhar-front, aadhar-back, rejected documents, PAN card

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
            self,
            data_dir: str = "data/",
            train_val_test_split_ratio: tuple = (0.8, 0.2),
            batch_size: int = 8,
            sampler: str = "random",
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        """Initialize a `DocumentsDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split_ratio: The ratio of split for train, val, test sets. Defaults to `(0.8, 0.2)`.
        :param batch_size: The batch size. Defaults to `8`.
        :param sampler: The train sampler to use. Defaults to `random`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[ImageFolder] = None
        self.data_val: Optional[ImageFolder] = None
        self.data_test: Optional[ImageFolder] = None
        self.data_predict: Optional[ImageFolder] = None
        self.batch_size_per_device = batch_size

        self.train_transforms = v2.Compose([
            v2.Resize(size=518, interpolation=InterpolationMode.BICUBIC),
            v2.CenterCrop(size=(518, 518)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=torch.Tensor([0.4850, 0.4560, 0.4060]), std=torch.Tensor([0.2290, 0.2240, 0.2250]))
        ])

        self.val_transforms = v2.Compose([
            v2.Resize(size=518, interpolation=InterpolationMode.BICUBIC),
            v2.CenterCrop(size=(518, 518)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=torch.Tensor([0.4850, 0.4560, 0.4060]), std=torch.Tensor([0.2290, 0.2240, 0.2250]))
        ])

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of Document classes (4).
        """
        return 4

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        original_dataset = osp.join(self.hparams.data_dir, 'Documents')
        train_path = osp.join(self.hparams.data_dir, 'train')
        val_path = osp.join(self.hparams.data_dir, 'val')

        # create train, val splits
        if (not osp.exists(train_path)) or (not osp.exists(val_path)):
            log.info(f"Creating train, val splits with ratio {self.hparams.train_val_test_split_ratio}")
            splitfolders.ratio(original_dataset, output=self.hparams.data_dir,
                               ratio=self.hparams.train_val_test_split_ratio,
                               move='symlink')

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        original_dataset = osp.join(self.hparams.data_dir, 'Documents')
        train_path = osp.join(self.hparams.data_dir, 'train')
        val_path = osp.join(self.hparams.data_dir, 'val')
        if stage == 'fit':
            self.data_train = ImageFolder(train_path, self.train_transforms)
            self.data_val = ImageFolder(val_path, self.val_transforms)
        if stage == 'test':
            self.data_test = ImageFolder(original_dataset, self.val_transforms)
        if stage == 'predict':
            self.data_predict = ImageFolderWithPaths(train_path, transform=self.val_transforms)

    def get_sampler(self):
        """fetches the appropriate sampler to use for the train_dataloader"""
        if self.hparams.sampler == 'random':
            return torch.utils.data.RandomSampler(data_source=self.data_train)
        elif self.hparams.sampler == 'imbalanced':
            ImbalancedDatasetSampler(self.data_train)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=self.get_sampler(),
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        return DataLoader(
            dataset=self.data_predict,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = DocumentsDataModule()
