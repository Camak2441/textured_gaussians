import torch


class Quadratic1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * x).mean()


class Quadratic01Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x - 0.5
        return 4 * (y * y).mean()


class TriangleQuadraticLoss(torch.nn.Module):
    def __init__(self, valley: float):
        super().__init__()
        self.v = 1 - valley
        self.iv2 = 1 / (self.v * self.v)
        self.m = 1 / (1 - 2 * self.v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + (self.v - 1)
        return torch.where(x < 1 - 2 * self.v, x * self.m, self.iv2 * (y * y)).mean()


class HalfQuadraticLoss(torch.nn.Module):
    def __init__(self, valley: float):
        super().__init__()
        self.v = 1 - valley
        self.iv2 = 1 / (self.v * self.v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + (self.v - 1)
        return torch.where(x < self.v, torch.zeros_like(x), self.iv2 * (y * y)).mean()
