import torch
import torch.nn as nn
import torch.nn.functional as F

from .scalenorm import ScaleNorm
from .powernorm import MaskPowerNorm

from einops import rearrange, repeat


def Flatten(inp: torch.Tensor) -> torch.Tensor:
    # spatiotemporal flattening of (batch, length, dim) into (batch, length x dim)
    out = rearrange(inp, "batch len dy -> batch (dy len) 1")
    return out


def Localize(inp: torch.Tensor, variables: int) -> torch.Tensor:
    # split spatiotemporal into individual vars and fold into batch dim
    return rearrange(
        inp,
        "batch (variables len) dim -> (variables batch) len dim",
        variables=variables,
    )


def MakeSelfMaskFromSeq(seq_mask: torch.Tensor):
    if seq_mask is None:
        return None
    batch, length, dim = seq_mask.shape
    assert dim == 1
    mask_rows = repeat(seq_mask, f"batch len 1 -> batch {length} len")
    mask_cols = repeat(seq_mask, f"batch len 1 -> batch len {length}")
    mask = torch.max(mask_rows, mask_cols).bool()
    return mask


def MakeCrossMaskFromSeq(self_seq_mask: torch.Tensor, cross_seq_mask: torch.Tensor):
    if self_seq_mask is None:
        return None

    batch_, cross_len, dim = cross_seq_mask.shape
    assert dim == 1
    batch, self_len, dim = self_seq_mask.shape
    assert batch_ == batch
    assert dim == 1

    mask_cols = repeat(self_seq_mask, f"batch len 1 -> batch len {cross_len}")
    mask_rows = repeat(cross_seq_mask, f"batch len 1 -> batch {self_len} len")
    mask = torch.max(mask_rows, mask_cols).bool()
    return mask


def WindowTime(
    inp: torch.Tensor, dy: int, windows: int, window_offset: int
) -> torch.Tensor:
    # stack
    if windows == 1 or inp is None:
        return inp
    x = rearrange(inp, "batch (dy len) dim -> batch len dy dim", dy=dy)

    if window_offset:
        # shift
        b, l, _, dim = x.shape
        window_len = l // 2
        shift_by = window_len // window_offset
        x = torch.roll(x, -shift_by, dims=1)

    # window and flatten
    x = rearrange(
        x, "batch (windows len) dy dim -> (batch windows) (dy len) dim", windows=windows
    )
    return x


def ReverseWindowTime(
    inp: torch.Tensor, dy: int, windows: int, window_offset: int
) -> torch.Tensor:
    if windows == 1 or inp is None:
        return inp
    # reverse window and stack
    x = rearrange(
        inp,
        "(batch windows) (dy len) dim -> batch (windows len) dy dim",
        dy=dy,
        windows=windows,
    )

    if window_offset:
        # shift
        b, l, _, dim = x.shape
        window_len = l // 2
        shift_by = window_len // window_offset
        x = torch.roll(x, shift_by, dims=1)

    # flatten
    x = rearrange(x, "batch len dy dim -> batch (dy len) dim", dy=dy)
    return x


def ReverseLocalize(inp: torch.Tensor, variables: int) -> torch.Tensor:
    return rearrange(
        inp,
        "(variables batch) len dim -> batch (variables len) dim",
        variables=variables,
    )


def ShiftBeforeWindow(inp: torch.Tensor, windows: int, offset: int = 2):
    # SWIN Transformer style window offsets
    b, l, v, d = inp.shape
    window_len = l // windows
    shift_by = window_len // offset
    return torch.roll(inp, -shift_by, dims=1)


def ReverseShiftBeforeWindow(inp: torch.Tensor, windows: int, offset: int = 2):
    b, l, v, d = inp.shape
    window_len = l // windows
    shift_by = window_len // offset
    return torch.roll(inp, shift_by, dims=1)


def Stack(inp: torch.Tensor, dy: int):
    return rearrange(inp, "batch (dy len) dim -> batch len dy dim", dy=dy)


def FoldForPred(inp: torch.Tensor, dy: int) -> torch.Tensor:
    out = rearrange(inp, "batch (dy len) dim -> dim batch len dy", dy=dy)
    out = out.squeeze(0)
    return out


class ConvBlock(nn.Module):
    def __init__(
        self,
        split_length_into,
        d_model,
        conv_kernel_size=3,
        conv_stride=1,
        pool=True,
        pool_kernel_size=3,
        pool_stride=2,
        activation="gelu",
    ):
        super().__init__()
        self.split_length = split_length_into
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(d_model)

        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unrecognized ConvBlock activation: `{activation}`")

        self.pool = (
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=1)
            if pool
            else lambda x: x
        )

    def conv_forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

    def forward(self, x):
        x = rearrange(
            x, f"batch (sl len) d_model -> (batch sl) d_model len", sl=self.split_length
        )
        x = self.conv_forward(x)
        x = rearrange(
            x, f"(batch sl) d_model len -> batch (sl len) d_model", sl=self.split_length
        )
        return x


class Normalization(nn.Module):
    def __init__(self, method, d_model=None):
        super().__init__()
        assert method in ["layer", "scale", "batch", "power", "none"]
        if method == "layer":
            assert d_model
            self.norm = nn.LayerNorm(d_model)
        elif method == "scale":
            self.norm = ScaleNorm(d_model)
        elif method == "power":
            self.norm = MaskPowerNorm(d_model, warmup_iters=1000)
        elif method == "none":
            self.norm = lambda x: x
        else:
            assert d_model
            self.norm = nn.BatchNorm1d(d_model)
        self.method = method

    def forward(self, x):
        if self.method == "batch":
            return self.norm(x.transpose(-1, 1)).transpose(-1, 1)
        return self.norm(x)
