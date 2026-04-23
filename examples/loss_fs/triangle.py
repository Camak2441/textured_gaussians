import torch


class TriangleLoss(torch.nn.Module):
    def __init__(self, peak: float):
        super().__init__()
        self.peak = peak

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l1 = x / self.peak
        l2 = (x - 1) / (self.peak - 1)
        return torch.where(x < self.peak, l1, l2).mean()


class Triangle0Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1 - x).mean()


class Triangle1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()
