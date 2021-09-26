from dataclasses import dataclass

import torch


@dataclass
class Task:
    x_context: torch.Tensor
    y_context: torch.Tensor
    x_target: torch.Tensor
    y_target: torch.Tensor
    task_info: str
