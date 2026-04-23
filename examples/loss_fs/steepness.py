import torch


class SteepnessLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, scales: torch.Tensor, steepnesses: torch.Tensor):
        sizes = torch.exp(scales).amax(dim=1)
        target_steepnesses = torch.log(sizes + torch.e) + 10
        loss = torch.log(
            1 + torch.nn.functional.softplus(steepnesses / target_steepnesses - 1)
        )
        return loss.mean()
