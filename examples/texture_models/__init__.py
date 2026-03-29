import json
from texture_models.utils import add_dict, check_keys_are_in_order, quote_json
from texture_models.mlp import MLP, FourierMLP, FourierSelector
from texture_models.siren import SIREN

MODEL_PREFIXES = {
    "FourierMLP": FourierMLP,
    "MLP": MLP,
    "SIREN": SIREN,
    "FourierSelector": FourierSelector,
}


GET_SUBMODEL_ARGS = {
    "FourierSelector": {},
}


def _dump_argstr(kwargs):
    match kwargs:
        case bool():
            return str(kwargs).lower()
        case int():
            return str(kwargs)
        case float():
            return str(kwargs)
        case str():
            return '"' + kwargs + '"'
        case list():
            s = []
            for item in kwargs:
                s.append(_dump_argstr(item))
            return "[" + ",".join(s) + "]"
        case dict():
            s = []
            for key in kwargs:
                s.append(key + "=" + _dump_argstr(kwargs[key]))
            return "(" + ",".join(s) + ")"


def _load_argstr(argstr: str):
    return json.loads(quote_json("{" + argstr[1:-1].replace("=", ":") + "}"))


def canonical_model_name(model_name: str):
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                print("Argstr does not contain args in brackets. Continuing.")
                continue
            kwargs = _load_argstr(argstr)
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
            return prefix + _dump_argstr(result_kwargs)
    raise Exception(f"Unknown model {model_name}")


def load_model(model_name: str, over_kwargs={}):
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                print("Argstr does not contain args in brackets. Continuing.")
                continue
            kwargs = _load_argstr(argstr)
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
