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

import math
import torch
from typing import Optional
from torch.optim import Optimizer

from .lr_scheduler import LearningRateScheduler


class TransformerLRScheduler(LearningRateScheduler):
    r"""
    Transformer Learning Rate Scheduler proposed in "Attention Is All You Need"

    Args:
        optimizer (Optimizer): Optimizer.
        init_lr (float): Initial learning rate.
        peak_lr (float): Maximum learning rate.
        final_lr (float): Final learning rate.
        final_lr_scale (float): Final learning rate scale
        warmup_steps (int): Warmup the learning rate linearly for the first N updates
        decay_steps (int): Steps in decay stages
    """

    def __init__(
        self,
        optimizer: Optimizer,
        init_lr: float,
        peak_lr: float,
        final_lr: float,
        final_lr_scale: float,
        warmup_steps: int,
        decay_steps: int,
    ) -> None:
        assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"
        assert isinstance(decay_steps, int), "total_steps should be inteager type"

        super(TransformerLRScheduler, self).__init__(optimizer, init_lr)
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        self.warmup_rate = self.peak_lr / self.warmup_steps
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.init_lr = init_lr
        self.update_steps = 0

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps

        if (
            self.warmup_steps
            <= self.update_steps
            < self.warmup_steps + self.decay_steps
        ):
            return 1, self.update_steps - self.warmup_steps

        return 2, None

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        self.update_steps += 1
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.update_steps * self.warmup_rate
        elif stage == 1:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 2:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)

        return self.lr
