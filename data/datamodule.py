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
            self.datasetCls(split=split),
            shuffle=shuffle,
            num_workers=12,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
