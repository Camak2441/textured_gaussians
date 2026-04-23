import torch


def gen_lin_seq(
    in_dim,
    out_dim,
    out_activation,
    hidden_dims=[],
    hidden_activation=torch.nn.ReLU(),
    initializer=torch.nn.init.zeros_,
):
    try:
        len(hidden_activation)
    except TypeError:
        hidden_activation = [hidden_activation] * len(hidden_dims)
    if len(hidden_dims) == 0:
        layer = torch.nn.Linear(in_dim, out_dim)
        initializer(layer.weight)
        return torch.nn.Sequential(
            layer,
            out_activation,
        )
    hidden_layers = []
    for i in range(len(hidden_dims) - 1):
        layer = torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1])
        initializer(layer.weight)
        hidden_layers.append(layer)
        hidden_layers.append(hidden_activation[i + 1])
    in_layer = torch.nn.Linear(in_dim, hidden_dims[0])
    out_layer = torch.nn.Linear(hidden_dims[-1], out_dim)
    initializer(in_layer.weight)
    initializer(out_layer.weight)
    return torch.nn.Sequential(
        in_layer,
        hidden_activation[0],
        *hidden_layers,
        out_layer,
        out_activation,
    )
