# MIT License
#
# Copyright (c) 2021 Soohwan Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from typing import Optional

from .lr_scheduler import LearningRateScheduler
from .reduce_lr_on_plateau_lr_scheduler import ReduceLROnPlateauScheduler
from .warmup_lr_scheduler import WarmupLRScheduler


class WarmupReduceLROnPlateauScheduler(LearningRateScheduler, ReduceLROnPlateau):
    r"""
    Warmup learning rate until `warmup_steps` and reduce learning rate on plateau after.

    Args:
        optimizer (Optimizer): wrapped optimizer.
        init_lr (float): Initial learning rate.
        peak_lr (float): Maximum learning rate.
        warmup_steps (int): Warmup the learning rate linearly for the first N updates.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        init_lr: float,
        peak_lr: float,
        warmup_steps: int,
        patience: int = 1,
        factor: float = 0.3,
    ) -> None:
        super(WarmupReduceLROnPlateauScheduler, self).__init__(optimizer, init_lr)
        self.warmup_steps = warmup_steps
        self.update_steps = 0
        self.warmup_rate = (
            (peak_lr - init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        )
        self.schedulers = [
            WarmupLRScheduler(
                optimizer=optimizer,
                init_lr=init_lr,
                peak_lr=peak_lr,
                warmup_steps=warmup_steps,
            ),
            ReduceLROnPlateauScheduler(
                optimizer=optimizer,
                lr=peak_lr,
                patience=patience,
                factor=factor,
            ),
        ]

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        return {}

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps
        else:
            return 1, None

    def step(self, val_loss: Optional[float] = None, is_end_epoch=False):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.schedulers[0].step()
        elif stage == 1 and is_end_epoch:
            self.schedulers[1].step(val_loss)

        self.update_steps += 1

        return self.get_lr()
