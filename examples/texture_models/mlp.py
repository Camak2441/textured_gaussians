import torch
from texture_models.utils import gen_lin_seq
from texture_models.fourier import FourierFeatures


class MLP(torch.nn.Module):
    def __init__(self, in_dim=3, hidden_dims=[256, 256, 256], out_dim=4):
        super().__init__()
        self.net = gen_lin_seq(
            in_dim,
            out_dim,
            torch.nn.Sigmoid(),
            hidden_dims=hidden_dims,
            initializer=lambda weight: torch.nn.init.kaiming_uniform(weight),
        )

    def forward(self, x):
        return self.net(x)


class FourierMLP(torch.nn.Module):
    def __init__(
        self,
        in_dim=3,
        num_frequencies=256,
        sigma=1.0,
        hidden_dims=[256, 256, 256],
        out_dim=4,
    ):
        super().__init__()
        self.fourier = FourierFeatures(in_dim, num_frequencies, sigma)
        self.net = gen_lin_seq(
            self.fourier.out_dim,
            out_dim,
            torch.nn.Sigmoid(),
            hidden_dims=hidden_dims,
            initializer=lambda weight: torch.nn.init.kaiming_uniform(weight),
        )

    def forward(self, x):
        return self.net(self.fourier(x))


class FourierSelector(torch.nn.Module):
    def __init__(
        self,
        in_dim=3,
        num_freqs=15,
        hidden_dims=[32],
        sub_models=[],
    ):
        super().__init__()
        assert in_dim == 3, "FourierSelector2 not compatible with in_dim != 3"
        self.num_freqs = num_freqs
        self.sub_models = sub_models
        for i in range(len(self.sub_models)):
            self.register_module(f"sub_models[{i}]", self.sub_models[i])
        self.net = gen_lin_seq(
            num_freqs + 2,
            len(sub_models),
            torch.nn.Softmax(dim=1),
            hidden_dims=hidden_dims,
            initializer=lambda weight: torch.nn.init.kaiming_uniform(weight),
        )

    def forward(self, x):
        x0 = x[:, 0:1] * torch.pi
        encoded = torch.cat(
            [
                x[:, 1:3],
                *(torch.sin(x0 * (1 << freq)) for freq in range(self.num_freqs)),
            ],
            dim=1,
        )
        model_weights = self.net(encoded)
        model_results = torch.stack(
            [
                self.sub_models[i](x) * model_weights[:, i : i + 1]
                for i in range(len(self.sub_models))
            ],
            dim=-1,
        )
        return torch.sum(model_results, dim=-1)


class FourierSelector2(torch.nn.Module):
    def __init__(
        self,
        in_dim=3,
        num_freqs=15,
        hidden_dims=[32],
        sub_models=[],
    ):
        super().__init__()
        assert in_dim == 3, "FourierSelector2 not compatible with in_dim != 3"
        self.num_freqs = num_freqs
        self.sub_models = sub_models
        for i in range(len(self.sub_models)):
            self.register_module(f"sub_models[{i}]", self.sub_models[i])
        self.net = gen_lin_seq(
            num_freqs + 2,
            len(sub_models),
            torch.nn.Sigmoid(),
            hidden_dims=hidden_dims,
            initializer=lambda weight: torch.nn.init.kaiming_uniform(weight),
        )

    def forward(self, x):
        x0 = x[:, 0:1] * torch.pi
        encoded = torch.cat(
            [
                x[:, 1:3],
                *(torch.sin(x0 * (1 << freq)) for freq in range(self.num_freqs)),
            ],
            dim=1,
        )
        model_weights = self.net(encoded)
        model_results = torch.stack(
            [
                self.sub_models[i](x) * model_weights[:, i : i + 1]
                + (1 - model_weights[:, i : i + 1])
                for i in range(len(self.sub_models))
            ],
            dim=-1,
        )
        return torch.prod(model_results, dim=-1)
