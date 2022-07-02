import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from einops import rearrange


class MNISTDset(Dataset):
    def __init__(self, context_points: int, split: str):
        transform = transforms.Compose([transforms.ToTensor()])
        self.dset = torchvision.datasets.MNIST(
            root="./images/", train=split == "train", download=True, transform=transform
        )
        self.context_points = context_points

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, i):
        img, label = self.dset[i]
        y = rearrange(img, "1 h w -> w h")
        y_c = y[: self.context_points]
        y_t = y[self.context_points :]

        x = torch.arange(28).view(-1, 1).float() / 28
        x_c = x[: self.context_points]
        x_t = x[self.context_points :]

        return x_c, y_c, x_t, y_t

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--context_points", type=int, default=20)
