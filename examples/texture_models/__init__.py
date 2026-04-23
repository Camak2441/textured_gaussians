import torch
from examples.clsarg_utils import (
    add_dict,
    check_keys_are_in_order,
    dump_argstr,
    load_argstr,
    quote_json,
)
from texture_models.mlp import MLP, FourierMLP, FourierSelector, FourierSelector2
from texture_models.siren import SIREN
from texture_models.debug import ConstColor

MODEL_PREFIXES = {
    "FourierMLP": FourierMLP,
    "MLP": MLP,
    "SIREN": SIREN,
    "FourierSelector": FourierSelector,
    "FourierSelector2": FourierSelector2,
    "ConstColor": ConstColor,
}


GET_DEFAULT_ARGS = {
    "FourierSelector": {"input_type": "gaussian"},
    "FourierSelector2": {"input_type": "gaussian"},
}


GET_SUBMODEL_ARGS = {
    "FourierSelector": {"sub_models": {"in_dim": 3}},
    "FourierSelector2": {"sub_models": {"in_dim": 3}},
}


def canonical_model_name(model_name: str) -> str:
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                print("Argstr does not contain args in brackets. Continuing.")
                continue
            kwargs = load_argstr(argstr)
            if prefix in GET_DEFAULT_ARGS:
                kwargs = add_dict(GET_DEFAULT_ARGS[prefix], kwargs)
            args = list(kwargs)
            args.sort()
            result_kwargs = {}
            for arg in args:
                if arg.endswith("model"):
                    result_kwargs[arg] = canonical_model_name(kwargs[arg])
                elif arg.endswith("models"):
                    result_kwargs[arg] = [
                        canonical_model_name(name) for name in kwargs[arg]
                    ]
                else:
                    result_kwargs[arg] = kwargs[arg]
            return prefix + dump_argstr(result_kwargs)
    raise Exception(f"Unknown model {model_name}")


def pop_arg_from_model_name(model_name: str, arg: str) -> tuple[str, any]:
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                print("Argstr does not contain args in brackets. Continuing.")
                continue
            kwargs = load_argstr(argstr)
            if prefix in GET_DEFAULT_ARGS:
                kwargs = add_dict(GET_DEFAULT_ARGS[prefix], kwargs)
            if arg in kwargs:
                value = kwargs.pop(arg)
                return prefix + dump_argstr(kwargs), value
            else:
                return model_name, None


def add_arg_to_model_name(model_name: str, arg: str, value: any) -> str:
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                print("Argstr does not contain args in brackets. Continuing.")
                continue
            kwargs = load_argstr(argstr)
            if prefix in GET_DEFAULT_ARGS:
                kwargs = add_dict(GET_DEFAULT_ARGS[prefix], kwargs)
            kwargs[arg] = value
            return prefix + dump_argstr(kwargs)


def load_model(model_name: str, over_kwargs={}) -> torch.nn.Module:
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
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
            for arg in kwargs:
                if arg.endswith("model"):
                    sub_model_kwargs = {}
                    if arg in GET_SUBMODEL_ARGS[prefix]:
                        sub_model_kwargs = GET_SUBMODEL_ARGS[prefix][arg](**kwargs)
                    kwargs[arg] = load_model(kwargs[arg], sub_model_kwargs)
                if arg.endswith("models"):
                    sub_models = []
                    for sub_model_arg in kwargs[arg]:
                        sub_model_kwargs = {}
                        if arg in GET_SUBMODEL_ARGS[prefix]:
                            sub_model_kwargs = GET_SUBMODEL_ARGS[prefix][arg](**kwargs)
                        sub_models.append(load_model(sub_model_arg, sub_model_kwargs))
                    kwargs[arg] = sub_models
            return MODEL_PREFIXES[prefix](**kwargs)
    raise Exception(f"Unknown model {model_name}")
