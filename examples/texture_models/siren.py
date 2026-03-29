import math
import torch


class SIRENLayer(torch.nn.Module):
    """A single SIREN layer: sin(omega_0 * (W x + b)).

    Initialization follows Sitzmann et al. 2020:
      - First layer:  W ~ U(-1/in_dim,  1/in_dim)
      - Other layers: W ~ U(-sqrt(6/in_dim)/omega_0, sqrt(6/in_dim)/omega_0)
    """

    def __init__(
        self, in_dim: int, out_dim: int, omega_0: float = 30.0, is_first: bool = False
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self._init_weights(in_dim, is_first)

    def _init_weights(self, in_dim: int, is_first: bool):
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_dim
            else:
                bound = math.sqrt(6.0 / in_dim) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(torch.nn.Module):
    """Sinusoidal Representation Network (Sitzmann et al. 2020).

    Stacks SIRENLayer hidden layers then a linear output layer with Sigmoid.
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dims: list = [256, 256, 256],
        out_dim: int = 4,
        omega_0: float = 30.0,
        hidden_omegas: float | list[float] = 1.0,
    ):
        super().__init__()
        layers = []

        dims = [in_dim] + list(hidden_dims)
        if isinstance(hidden_omegas, list):
            assert len(hidden_omegas) + 1 == len(hidden_dims)
            omegas = [omega_0] + hidden_omegas
        else:
            omegas = [omega_0] + [hidden_omegas] * (len(hidden_dims) - 1)
        for i in range(len(dims) - 1):
            layers.append(
                SIRENLayer(dims[i], dims[i + 1], omega_0=omegas[i], is_first=(i == 0))
            )

        out_layer = torch.nn.Linear(dims[-1], out_dim)
        torch.nn.init.zeros_(out_layer.weight)
        layers.append(out_layer)
        layers.append(torch.nn.Sigmoid())

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
