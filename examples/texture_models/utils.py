import re
from typing import Any, Dict
import torch


def add_dict(d1, d2):
    dres = d1.copy()
    for k in d2:
        if k not in d1:
            dres[k] = d2[k]
    return dres


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


def check_keys_are_in_order(kwargs: Dict[str, Any]):
    prev = ""
    for key in kwargs:
        if prev > key:
            return False
        prev = key
    return True


def quote_json_val(s):
    val = s.strip()
    if re.fullmatch(r"0|[1-9][0-9]*|(0|[1-9][0-9]*)\.[0-9]*|true|false", val):
        return val
    if val.startswith('"') and val.endswith('"'):
        return val
    if val == "":
        return val
    return '"' + val + '"'


def quote_json_key(s):
    key = s.strip()
    if key.startswith('"') and key.endswith('"'):
        return key
    return '"' + key + '"'


def quote_json(s):
    bracket_stack = []
    segs = []

    index = 0
    seg = []

    def get_seg():
        nonlocal seg
        seg_s = "".join(seg)
        seg.clear()
        return seg_s

    while index < len(s):
        c = s[index]
        index += 1
        if len(bracket_stack) > 0 and bracket_stack[-1] == '"':
            seg.append(c)
            match c:
                case '"':
                    if not (
                        len(seg) > 0
                        and seg[-1] == "\\"
                        and (len(seg) < 2 or seg[-2] != "\\")
                    ):
                        bracket_stack.pop()
                        seg_s = get_seg()
                        segs.append(seg_s)
        else:
            match c:
                case '"':
                    seg_s = get_seg()
                    segs.append(seg_s)
                    seg.append('"')
                    bracket_stack.append('"')
                case "{":
                    seg_s = get_seg()
                    assert (
                        seg_s.strip() == ""
                    ), f"Unexpected value before {index} in {s}"
                    segs.append("{")
                    bracket_stack.append("{")
                case "}":
                    seg_s = get_seg()
                    assert (
                        len(bracket_stack) > 0
                    ), f"Unmatched bracket at {index} in {s}"
                    if bracket_stack[-1] == ":":
                        quoted = quote_json_val(seg_s)
                        assert (
                            quoted is not None
                        ), f"Missing value before {index} in {s}"
                        segs.append(quoted + "}")
                        bracket_stack.pop()
                    else:
                        assert (
                            seg_s.strip() == ""
                        ), f"Unexpected value before {index} in {s}"
                    assert (
                        bracket_stack[-1] == "{"
                    ), f"Unmatched bracket at {index} in {s}"
                    bracket_stack.pop()
                case "[":
                    seg_s = get_seg()
                    assert (
                        seg_s.strip() == ""
                    ), f"Unexpected value before {index} in {s}"
                    segs.append("[")
                    bracket_stack.append("[")
                case "]":
                    seg_s = get_seg()
                    assert (
                        len(bracket_stack) > 0
                    ), f"Unmatched bracket at {index} in {s}"
                    if bracket_stack[-1] == ",":
                        quoted = quote_json_val(seg_s)
                        assert (
                            quoted is not None
                        ), f"Missing value before {index} in {s}"
                        segs.append(quoted + "]")
                        bracket_stack.pop()
                    else:
                        quoted = quote_json_val(seg_s)
                        if quoted is not None:
                            segs.append(quoted + "]")

                    assert (
                        bracket_stack[-1] == "["
                    ), f"Unmatched bracket at {index} in {s}"
                    bracket_stack.pop()
                case ":":
                    seg_s = get_seg()
                    quoted = quote_json_key(seg_s)
                    assert quoted is not None, f"Missing value before {index} in {s}"
                    assert len(bracket_stack) > 0, f"Unmatched colon at {index} in {s}"
                    segs.append(quoted + ":")
                    if bracket_stack[-1] == ",":
                        bracket_stack.pop()
                    assert (
                        bracket_stack[-1] in "{"
                    ), f"Unmatched colon at {index} in {s}"
                    bracket_stack.append(":")
                case ",":
                    seg_s = get_seg()
                    quoted = quote_json_val(seg_s)
                    assert quoted is not None, f"Missing value before {index} in {s}"
                    segs.append(quoted + ",")
                    assert len(bracket_stack) > 0, f"Unexpected comma at {index} in {s}"
                    assert (
                        bracket_stack[-1] in "[:,"
                    ), f"Unexpected comma at {index} in {s}"
                    if bracket_stack[-1] == ":":
                        bracket_stack.pop()
                    if bracket_stack[-1] == ",":
                        bracket_stack.pop()
                        assert (
                            bracket_stack[-1] in "["
                        ), f"Unexpected comma at {index} in {s}"
                    bracket_stack.append(",")
                case _:
                    seg.append(c)

    seg_s = get_seg()
    if len(segs) != 0:
        assert seg_s.strip() == "", f"Unexpected value before {index} in {s}"
    else:
        quoted = quote_json_val(seg_s)
        assert quoted is not None, f"Missing value before {index} in {s}"
        segs.append(quoted)

    return "".join(segs)
