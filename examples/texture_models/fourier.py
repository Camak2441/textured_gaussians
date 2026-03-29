import math
from numbers import Number
from typing import List, Union
import torch


class FourierFeatures(torch.nn.Module):

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int = 256,
        sigma: Union[float, List[float]] = 1.0,
        frequencies=None,
    ):
        super().__init__()
        if frequencies is None:
            if isinstance(sigma, Number):
                B = torch.randn(in_dim, num_frequencies) * sigma
            else:
                assert in_dim == len(
                    sigma
                ), f"Expected {in_dim} features, received {len(sigma)}."
                B = torch.stack(
                    [torch.randn(num_frequencies) * s for s in sigma], dim=0
                )
        else:
            B = frequencies
        self.register_buffer("B", B)
        self.out_dim = 2 * num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = (2 * math.pi * x) @ self.B
        return torch.cat([proj.sin(), proj.cos()], dim=-1)
