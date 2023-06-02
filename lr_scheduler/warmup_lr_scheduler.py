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

import torch
from typing import Optional
from torch.optim import Optimizer

from .lr_scheduler import LearningRateScheduler


class WarmupLRScheduler(LearningRateScheduler):
    """
    Warmup learning rate until `total_steps`

    Args:
        optimizer (Optimizer): wrapped optimizer.

    """

    def __init__(
        self,
        optimizer: Optimizer,
        init_lr: float,
        peak_lr: float,
        warmup_steps: int,
    ) -> None:
        super(WarmupLRScheduler, self).__init__(optimizer, init_lr)
        self.init_lr = init_lr
        if warmup_steps != 0:
            warmup_rate = peak_lr - init_lr
            self.warmup_rate = warmup_rate / warmup_steps
        else:
            self.warmup_rate = 0
        self.update_steps = 1
        self.lr = init_lr
        self.warmup_steps = warmup_steps

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        if self.update_steps < self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_steps
            self.set_lr(self.optimizer, lr)
            self.lr = lr
        self.update_steps += 1
        return self.lr
