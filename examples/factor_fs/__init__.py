from examples.factor_fs.fs import (
    Constant,
    Exponential,
    FactorFn,
    LinearInterpolate,
    Quadratic,
    SquareRoot,
)
from examples.clsarg_utils import (
    add_dict,
    check_keys_are_in_order,
    dump_argstr,
    load_argstr,
)

FACTOR_PREFIXES = {
    "Constant": Constant,
    "LinearInterpolate": LinearInterpolate,
    "Quadratic": Quadratic,
    "Exponential": Exponential,
    "ExpOneAtStep": Exponential.one_at_step_exponential,
    "SquareRoot": SquareRoot,
}


def canonical_factor_name(factor_name: str):
    for prefix in FACTOR_PREFIXES:
        if factor_name.startswith(prefix):
            argstr = factor_name[len(prefix) :]
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


def load_factor_fn(factor_name: str, over_kwargs={}) -> FactorFn:
    for prefix in FACTOR_PREFIXES:
        if factor_name.startswith(prefix):
            argstr = factor_name[len(prefix) :]
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
            return FACTOR_PREFIXES[prefix](**kwargs)
    raise Exception(f"Unknown factor {factor_name}")
