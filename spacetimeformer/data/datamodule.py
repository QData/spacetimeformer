import warnings

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int,
        collate_fn=None,
        overfit: bool = False,
    ):
        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size
        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]
        self.dataset_kwargs = dataset_kwargs
        self.workers = workers
        self.collate_fn = collate_fn
        if overfit:
            warnings.warn("Overriding val and test dataloaders to use train set!")
        self.overfit = overfit

    def train_dataloader(self, shuffle=True):
        return self._make_dloader("train", shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        return self._make_dloader("val", shuffle=shuffle)

    def test_dataloader(self, shuffle=False):
        return self._make_dloader("test", shuffle=shuffle)

    def _make_dloader(self, split, shuffle=False):
        if self.overfit:
            split = "train"
            shuffle = True
        return DataLoader(
            self.datasetCls(**self.dataset_kwargs, split=split),
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )
        parser.add_argument(
            "--overfit",
            action="store_true",
        )
