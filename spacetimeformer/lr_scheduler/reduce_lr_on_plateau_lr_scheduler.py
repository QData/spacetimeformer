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

from omegaconf import DictConfig
from torch.optim import Optimizer

from .lr_scheduler import LearningRateScheduler


class ReduceLROnPlateauScheduler(LearningRateScheduler):
    r"""
    Reduce learning rate when a metric has stopped improving. Models often benefit from reducing the learning rate by
    a factor of 2-10 once learning stagnates. This scheduler reads a metrics quantity and if no improvement is seen
    for a ‘patience’ number of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Optimizer.
        lr (float): Initial learning rate.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr: float,
        patience: int = 1,
        factor: float = 0.3,
    ) -> None:
        super(ReduceLROnPlateauScheduler, self).__init__(optimizer, lr)
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.val_loss = float("inf")
        self.count = 0
        self._old_val_loss = None

    def step(self, val_loss: float):
        if self.val_loss < val_loss:
            self.count += 1
            self.val_loss = val_loss
        else:
            self.count = 0
            self.val_loss = val_loss

        if self.patience == self.count:
            self.count = 0
            self.lr *= self.factor
            self.set_lr(self.optimizer, self.lr)

        return self.lr
