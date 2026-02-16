import torch
from texture_models.utils import gen_lin_seq


class MLP(torch.nn.Module):
    def __init__(self, in_dim=3, hidden_dims=[256, 256, 256], out_dim=4):
        super().__init__()
        self.net = gen_lin_seq(
            in_dim, out_dim, torch.nn.Sigmoid(), hidden_dims=hidden_dims
        )

    def forward(self, x):
        return self.net(x)
