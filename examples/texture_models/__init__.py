import json
from texture_models.utils import check_keys_are_in_order, quote_json
from texture_models.mlp import MLP

MODEL_PREFIXES = {
    "MLP": MLP,
}


def _load_argstr(argstr: str):
    return json.loads(quote_json("{" + argstr[1:-1].replace("=", ":") + "}"))


def canonical_model_name(model_name: str):
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                continue
            kwargs = _load_argstr(argstr)
            return (
                prefix
                + "("
                + "".join(
                    json.dumps(kwargs, separators=(",", "="), sort_keys=True).split()
                ).replace('"', "")[1:-1]
                + ")"
            )
    raise Exception(f"Unknown model {model_name}")


def load_model(model_name: str):
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                continue
            kwargs = _load_argstr(argstr)
            if not check_keys_are_in_order(kwargs):
                return
            if "tag" in kwargs:
                kwargs.pop("tag")
            return MODEL_PREFIXES[prefix](**kwargs)
    raise Exception(f"Unknown model {model_name}")
