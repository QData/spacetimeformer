import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, datasetCls, dataset_kwargs: dict, batch_size: int, workers: int):
        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size
        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]
        self.dataset_kwargs = dataset_kwargs
        self.workers = workers

    def train_dataloader(self):
        return self._make_dloader("train")

    def val_dataloader(self):
        return self._make_dloader("val")

    def test_dataloader(self):
        return self._make_dloader("test")

    def _make_dloader(self, split):
        return DataLoader(
            self.datasetCls(**self.dataset_kwargs, split=split),
            shuffle=True if split == "train" else False,
            batch_size=self.batch_size,
            num_workers=self.workers,
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
