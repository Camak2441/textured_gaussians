import math
from numbers import Number
from typing import List, Union
import torch


class ConstColor(torch.nn.Module):

    def __init__(
        self,
        out_color: List[float],
    ):
        super().__init__()
        self.register_buffer("out_color", torch.Tensor(out_color))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim]
        return self.out_color.repeat([x.size(0), 1])
