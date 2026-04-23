import torch

from examples.loss_fs.quadratic import (
    HalfQuadraticLoss,
    Quadratic01Loss,
    Quadratic1Loss,
    TriangleQuadraticLoss,
)
from examples.loss_fs.triangle import TriangleLoss, Triangle0Loss, Triangle1Loss
from examples.loss_fs.steepness import SteepnessLoss
from examples.clsarg_utils import (
    add_dict,
    check_keys_are_in_order,
    dump_argstr,
    load_argstr,
    quote_json,
)

LOSS_PREFIXES = {
    "Triangle": TriangleLoss,
    "Triangle0": Triangle0Loss,
    "Triangle1": Triangle1Loss,
    "Steepness": SteepnessLoss,
    "Quadratic01": Quadratic01Loss,
    "Quadratic1": Quadratic1Loss,
    "TriangleQuadratic": TriangleQuadraticLoss,
    "HalfQuadratic": HalfQuadraticLoss,
}


def canonical_loss_name(factor_name: str):
    for prefix in LOSS_PREFIXES:
        if factor_name.startswith(prefix):
            argstr = factor_name[len(prefix) :]
            if len(argstr) == 0:
                kwargs = {}
            else:
                if not (argstr.startswith("(") and argstr.endswith(")")):
                    print("Argstr does not contain args in brackets. Continuing.")
                    continue
                kwargs = load_argstr(argstr)
            args = list(kwargs)
            args.sort()
            result_kwargs = {}
            for arg in args:
                result_kwargs[arg] = kwargs[arg]
            return prefix + dump_argstr(result_kwargs)
    raise Exception(f"Unknown factor {factor_name}")


def load_loss_fn(loss_fn_name: str, over_kwargs={}) -> torch.nn.Module:
    for prefix in LOSS_PREFIXES:
        if loss_fn_name.startswith(prefix):
            argstr = loss_fn_name[len(prefix) :]
            if len(argstr) == 0:
                kwargs = {}
            else:
                if not (argstr.startswith("(") and argstr.endswith(")")):
                    print("Argstr does not contain args in brackets. Continuing.")
                    continue
                kwargs = load_argstr(argstr)
            kwargs = add_dict(over_kwargs, kwargs)

            if not check_keys_are_in_order(kwargs):
                print("Keys not in order. Continuing.")
                continue
            if "tag" in kwargs:
                kwargs.pop("tag")
            return LOSS_PREFIXES[prefix](**kwargs)
    raise Exception(f"Unknown loss_fn {loss_fn_name}")
