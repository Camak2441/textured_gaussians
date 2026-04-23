import json
import re
from typing import TypeVar

K1 = TypeVar("K1")
K2 = TypeVar("K2")
V1 = TypeVar("V1")
V2 = TypeVar("V2")


def add_dict(d1: dict[K1, V1], d2: dict[K2, V2]) -> dict[K1 | K2, V1 | V2]:
    dres = d1.copy()
    for k in d2:
        if k not in d1:
            dres[k] = d2[k]
    return dres


def check_keys_are_in_order(kwargs: dict[str, any]):
    prev = ""
    for key in kwargs:
        if prev > key:
            return False
        prev = key
    return True


def quote_json_val(s: str) -> str:
    val = s.strip()
    if re.fullmatch(r"0|[1-9][0-9]*|(0|[1-9][0-9]*)\.[0-9]*|true|false", val):
        return val
    if val.startswith('"') and val.endswith('"'):
        return val
    if val == "":
        return val
    return '"' + val + '"'


def quote_json_key(s: str) -> str:
    key = s.strip()
    if key.startswith('"') and key.endswith('"'):
        return key
    return '"' + key + '"'


def quote_json(s: str) -> str:
    bracket_stack: list[str] = []
    segs: list[str] = []

    index: int = 0
    seg: list[str] = []

    def get_seg() -> str:
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


def dump_argstr(kwargs: dict[str, any]) -> str:
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
                s.append(dump_argstr(item))
            return "[" + ",".join(s) + "]"
        case dict():
            s = []
            for key in kwargs:
                s.append(key + "=" + dump_argstr(kwargs[key]))
            return "(" + ",".join(s) + ")"


def load_argstr(argstr: str) -> dict[str, any]:
    if argstr in ("", "()"):
        return {}
    return json.loads(quote_json("{" + argstr[1:-1].replace("=", ":") + "}"))
