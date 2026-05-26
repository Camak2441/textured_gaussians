#!/usr/bin/env python3
"""
Compile a LaTeX table from evaluation results.

Usage:
  # No groups:
  python scripts/compile_latex_table.py \
    --models 2dgs tgs --datasets ns

  # Auto-group by init (Random / SfM), annotate group labels with splat counts,
  # and annotate model headers with tex_grad:
  python scripts/compile_latex_table.py \
    --models tgs tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_po_pswc08 \
             tgs_psfm tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_psfm_po_pswc08 \
    --group-by init \
    --group-params splats \
    --model-params tex_grad \
    --datasets ns

  # Explicit groups:
  python scripts/compile_latex_table.py \
    --groups "Random:tgs,tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_po_pswc08" \
             "SfM:tgs_psfm,tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_psfm_po_pswc08" \
    --datasets ns

Available metrics: psnr ssim lpips cvvdp render_time memory model_size
"""

LATEX_FONT_SIZES = [
    "tiny",
    "scriptsize",
    "footnotesize",
    "small",
    "normalsize",
    "large",
    "Large",
    "LARGE",
    "huge",
    "Huge",
]


def get_font_size_index(size: str | int) -> int:
    if isinstance(size, int):
        return size
    return LATEX_FONT_SIZES.index(size)


DEFAULT_VALUES = {
    "tex_grad": False,
    "opac_loss": False,
    "splats": 10000,
    "init": "Random",
    "tex_size": None,
}


MODEL_NAMES = {
    "2dgs": {
        "base": "2DGS",
        "tex_grad": False,
        "splats": 10000,
        "init": "Random",
        "tex_size": None,
    },
    "2dgs_oquad1-1000": {
        "base": "2DGS",
        "tex_grad": False,
        "opac_loss": True,
        "splats": 10000,
        "init": "Random",
        "tex_size": None,
    },
    "2dgs_g2000": {
        "base": "2DG-SS",
        "tex_grad": False,
        "splats": 2000,
        "init": "Random",
        "tex_size": None,
    },
    "2dgs_g2000_oquad1-1000": {
        "base": "2DG-SS",
        "tex_grad": False,
        "opac_loss": True,
        "splats": 2000,
        "init": "Random",
        "tex_size": None,
    },
    "2dgss_g1966_oquad1-1000_swc08": {
        "base": "2DG-SS",
        "tex_grad": False,
        "splats": 1966,
        "init": "Random",
        "tex_size": None,
    },
    "2dgss_g1966_swc08": {
        "base": "2DG-SS",
        "tex_grad": False,
        "splats": 1966,
        "init": "Random",
        "tex_size": None,
    },
    "2dgss_g1999_oquad1-1000_swc08": {
        "base": "2DG-SS",
        "tex_grad": False,
        "splats": 1999,
        "init": "Random",
        "tex_size": None,
    },
    "2dgss_g9833_swc08": {
        "base": "2DSS",
        "tex_grad": False,
        "splats": 9833,
        "init": "Random",
        "tex_size": None,
    },
    "2dgss_g9833_oquad1-1000_swc08": {
        "base": "2DG-SS",
        "tex_grad": False,
        "splats": 9833,
        "init": "Random",
        "tex_size": None,
    },
    "2dgss_g9999_oquad1-1000_swc08": {
        "base": "2DG-SS",
        "tex_grad": False,
        "splats": 9999,
        "init": "Random",
        "tex_size": None,
    },
    "2dgs_sfm": {
        "base": "2DGS",
        "tex_grad": False,
        "splats": 10000,
        "init": "SfM",
        "tex_size": None,
    },
    "2dgs_sfm_oquad1-1000": {
        "base": "2DGS",
        "tex_grad": False,
        "opac_loss": True,
        "splats": 10000,
        "init": "SfM",
        "tex_size": None,
    },
    "2dgss_g9833_sfm_oquad1-1000_swc08": {
        "base": "2DG-SS",
        "tex_grad": False,
        "splats": 9833,
        "init": "SfM",
        "tex_size": None,
    },
    "2dgss_g9833_sfm_swc08": {
        "base": "2DG-SS",
        "tex_grad": False,
        "splats": 9833,
        "init": "SfM",
        "tex_size": None,
    },
    "2dgss_g9999_sfm_oquad1-1000_swc08": {
        "base": "2DG-SS",
        "tex_grad": False,
        "splats": 9999,
        "init": "SfM",
        "tex_size": None,
    },
    "tgs": {
        "base": "TGS",
        "tex_grad": False,
        "splats": 10000,
        "init": "Random",
        "tex_size": 64,
    },
    "tgs_g2000": {
        "base": "TGS",
        "tex_grad": False,
        "splats": 2000,
        "init": "Random",
        "tex_size": 64,
    },
    "tgs_b2": {
        "base": "TGS",
        "tex_grad": True,
        "splats": 10000,
        "init": "Random",
        "tex_size": 64,
    },
    "tgs_b2_g2000": {
        "base": "TGS",
        "tex_grad": True,
        "splats": 2000,
        "init": "Random",
        "tex_size": 64,
    },
    "tgss4_b2_g1999_ot01-0_ott03-0_sgc02_swc08_po_pswc08": {
        "base": "TG-SS",
        "tex_grad": False,
        "splats": 1999,
        "init": "Random",
        "tex_size": 64,
    },
    "tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_po_pswc08": {
        "base": "TG-SS",
        "tex_grad": False,
        "splats": 9999,
        "init": "Random",
        "tex_size": 64,
    },
    "tgs_psfm": {
        "base": "TGS",
        "tex_grad": False,
        "splats": 10000,
        "init": "SfM",
        "tex_size": 64,
    },
    "tgs_b2_psfm": {
        "base": "TGS",
        "tex_grad": True,
        "splats": 10000,
        "init": "SfM",
        "tex_size": 64,
    },
    "tgs_b2_psfm_poquad1": {
        "base": "TGS",
        "opac_loss": True,
        "tex_grad": True,
        "splats": 10000,
        "init": "SfM",
        "tex_size": 64,
    },
    "tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_psfm_po_pswc08": {
        "base": "TG-SS",
        "tex_grad": False,
        "splats": 9999,
        "init": "SfM",
        "tex_size": 64,
    },
    "tgs_g2000_to0_tgs_b2_g2000_abp": {
        "base": "BilinearTGS",
        "tex_grad": False,
        "splats": 2000,
        "init": "Random",
        "tex_size": 64,
    },
    "tgs_to0_tgs_b2_psfm_abp": {
        "base": "BilinearTGS",
        "tex_grad": False,
        "splats": 10000,
        "init": "SfM",
        "tex_size": 64,
    },
    "mip_tgs_g2000_to0_tgs_b2_g2000_abp": {
        "base": "MipTGS",
        "tex_grad": False,
        "splats": 2000,
        "init": "Random",
        "tex_size": 64,
    },
    "mip_tgs_to0_tgs_b2_psfm_abp": {
        "base": "MipTGS",
        "tex_grad": False,
        "splats": 10000,
        "init": "SfM",
        "tex_size": 64,
    },
    "aniso_bilinear_tgs_g2000_to0_tgs_b2_g2000_abp": {
        "base": "AnisoTGS",
        "tex_grad": False,
        "splats": 2000,
        "init": "Random",
        "tex_size": 64,
    },
    "aniso_bilinear_tgs_to0_tgs_b2_psfm_abp": {
        "base": "AnisoTGS",
        "tex_grad": False,
        "splats": 10000,
        "init": "SfM",
        "tex_size": 64,
    },
    "tgs_b2_ta_t6": {
        "base": "TGS",
        "tex_grad": True,
        "splats": 10000,
        "init": "Random",
        "tex_size": 6,
    },
    "dtgs3_b2_ta_t8": {
        "base": "DTGS",
        "tex_grad": True,
        "splats": 10000,
        "init": "Random",
        "tex_size": 8,
    },
}


import argparse, json, os, glob, re

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Fields that can be used in --model-params / --group-params / --group-by
ANNOTATABLE_PARAMS = ["tex_grad", "opac_loss", "splats", "init"]


def _step_number(path):
    m = re.search(r"(\d+)\.(?:json|pt)$", path)
    return int(m.group(1)) if m else -1


def last_val_stats(results_dir, model, scene):
    pattern = os.path.join(results_dir, model, scene, "stats", "val_step*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=_step_number)
    with open(latest) as f:
        return json.load(f)


def last_mem_val_stats(results_dir, model, scene):
    pattern = os.path.join(results_dir, model, scene, "stats", "mem_val_step*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=_step_number)
    with open(latest) as f:
        return json.load(f)


def last_ckpt_size(results_dir, model, scene):
    """Return the size of the latest ckpt_<step>.pt file in bytes, or None."""
    pattern = os.path.join(results_dir, model, scene, "ckpts", "ckpt_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=_step_number)
    return os.path.getsize(latest)


# ---------------------------------------------------------------------------
# Metric definitions
# source: "val"  → val_step JSON field
#         "mem"  → mem_val_step JSON field
#         "ckpt" → ckpt file size in bytes (field ignored)
# ---------------------------------------------------------------------------


def optfold(n, s):
    return lambda opt: n if opt is None else s(opt)


def seq(*fs):
    def _seq(x):
        nonlocal fs
        last = None
        for f in fs:
            last = f(x)
        return last

    return _seq


def opt_map(f):
    return optfold(None, f)


def issome(opt):
    return opt is not None


def swap(t):
    return (t[1], t[0])


def proj(i):
    return lambda t: t[i]


def circ(f, g):
    return lambda x: f(g(x))


def ave(l):
    return sum(l) / len(l) if l else None


def color_cell(col):
    return lambda c: rf"\cellcolor{{{col}}}{c}"


def bold_cell():
    return lambda c: rf"\textbf{c}"


def italic_cell():
    return lambda c: rf"\textit{c}"


def underline_cell():
    return lambda c: rf"\underline{c}"


CELL_HIGHLIGHTS_PREFIXES = {
    "color": {"args": 1, "fn": color_cell},
    "bold": {"args": 0, "fn": bold_cell},
    "italic": {"args": 0, "fn": italic_cell},
    "underline": {"args": 0, "fn": underline_cell},
}


def get_cell_highlight(s: str):
    for prefix in CELL_HIGHLIGHTS_PREFIXES:
        if s.startswith(prefix):
            suffix = s[len(prefix) :]
            args = suffix.split("_")[1:]
            if len(args) != CELL_HIGHLIGHTS_PREFIXES[prefix]["args"]:
                continue
            return opt_map(CELL_HIGHLIGHTS_PREFIXES[prefix]["fn"](*args))
    assert False, f"Unknown highlight type {s}"


def _fmt_psnr(v):
    return f"{v:.2f}"


def _fmt_ssim(v):
    return f"{v:.3f}"


def _fmt_lpips(v):
    return f"{v:.3f}"


def _fmt_cvvdp(v):
    return f"{v:.3f}"


def _fmt_time(v):
    return f"{v * 1000:.1f}"  # seconds → ms


def _fmt_mem(v):
    return f"{v:.3f}"  # already in GB


def _fmt_size(v):
    return f"{v / 1e6:.2f}"  # bytes → MB


METRIC_DEFS = {
    "psnr": {
        "up_better": True,
        "label": "PSNR",
        "source": "val",
        "field": "psnr",
        "fmt": _fmt_psnr,
    },
    "ssim": {
        "label": "SSIM",
        "up_better": True,
        "source": "val",
        "field": "ssim",
        "fmt": _fmt_ssim,
    },
    "lpips": {
        "label": "LPIPS",
        "up_better": False,
        "source": "val",
        "field": "lpips",
        "fmt": _fmt_lpips,
    },
    "cvvdp": {
        "label": "CVVDP",
        "up_better": True,
        "source": "val",
        "field": "cvvdp_jod",
        "fmt": _fmt_cvvdp,
    },
    "render_time": {
        "label": "Time (ms)",
        "up_better": False,
        "source": "val",
        "field": "elapsed_time",
        "fmt": _fmt_time,
    },
    "memory": {
        "label": "Mem (GB)",
        "up_better": False,
        "source": "mem",
        "field": "mem",
        "fmt": _fmt_mem,
    },
    "model_size": {
        "label": "Size (MB)",
        "up_better": False,
        "source": "ckpt",
        "field": None,
        "fmt": _fmt_size,
    },
}

DEFAULT_METRICS = ["psnr", "ssim", "lpips"]

COMMAND_SCENES = {"midline", "average", "total_average"}


def _get_metric_value(metric_key, val_data, mem_data, ckpt_size):
    defn = METRIC_DEFS[metric_key]
    source = defn["source"]
    if source == "val":
        if val_data is None:
            return None
        return val_data.get(defn["field"])
    elif source == "mem":
        if mem_data is None:
            return None
        return mem_data.get(defn["field"])
    elif source == "ckpt":
        return ckpt_size
    return None


PARAM_LABELS = {
    "init": "Init",
    "splats": "Splats",
    "tex_grad": "TexGrad",
}


def _short_param_value(param, value):
    """Short value string for group header labels (no redundant unit)."""
    if param == "tex_grad":
        return r"$\partial$" if value else r"w/o $\partial$"
    if param == "opac_loss":
        return r"$\lopac$" if value else r"w/o $\lopac$"
    elif param == "splats":
        return str(value)
    return str(value)


def _format_param_value(param, value):
    """Format a single MODEL_NAMES field value into a display string."""
    if param == "tex_grad":
        return r"With TexGrad" if value else r"Without TexGrad"
    elif param == "splats":
        return f"{value} Splats"
    elif param == "init":
        return str(value)
    return str(value)


def get_model_display_name(model_key, model_params=None, model_name_overrides=None):
    """Build the display name for a model with optional parameter annotations.

    model_params selects which MODEL_NAMES fields to include:
      - "tex_grad" → appends " w/ TexGrad" to the base name when True
      - "splats"   → adds "N Splats" in parentheses
      - "init"     → adds "SfM" / "Random" in parentheses

    Example: get_model_display_name("tgs_b2_psfm", ["tex_grad", "splats", "init"])
             → "TGS w/ TexGrad (10000 Splats, SfM)"
    """
    if model_name_overrides and model_key in model_name_overrides:
        return model_name_overrides[model_key]
    info = MODEL_NAMES.get(model_key)
    if info is None:
        return model_key.replace("_", r"\_")
    name = info["base"]
    bracket_parts = []
    if model_params:
        for p in model_params:
            val = info.get(p, DEFAULT_VALUES[p])
            if val is None:
                continue
            if p == "tex_grad":
                if val:
                    name += r"$\partial$"
            elif p == "opac_loss":
                if val:
                    name += r"$\lopac$"
            else:
                bracket_parts.append(_format_param_value(p, val))
    if bracket_parts:
        name += f" ({', '.join(bracket_parts)})"
    return name


def get_group_display_name(group_label, group_models, group_params=None):
    """Build the display name for a group header, annotating with aggregated param values.

    For each param in group_params, collects all unique values across the group's
    models and appends them in parentheses. tex_grad is omitted when False.

    Example with group_params=["splats"]:
      group containing tgs (10000 splats) and tgss (9999 splats)
      → "Random (10000 Splats, 9999 Splats)"
    """
    if not group_params:
        return group_label
    all_value_strs = []
    for p in group_params:
        seen = set()
        for model_key in group_models:
            info = MODEL_NAMES.get(model_key, {})
            val = info.get(p)
            if val is None:
                continue
            if p == "tex_grad" and not val:
                continue  # only annotate when True
            v_str = _format_param_value(p, val)
            if v_str not in seen:
                seen.add(v_str)
                all_value_strs.append(v_str)
    if all_value_strs:
        return group_label + f" ({', '.join(all_value_strs)})"
    return group_label


def _parse_group_by_spec(spec):
    """Parse a single --group-by element into (field, value_groups).

    Simple form:   "init"
    → ("init", None)   # auto-derive groups from unique values

    Extended form: "splats:(2000,1966)=~2k:(10000,9833)=~10k"
    → ("splats", [([2000, 1966], "~2k"), ([10000, 9833], "~10k")])

    Values are coerced to int then float then kept as str.
    Group segments within the extended form are separated by ":" outside of "()".
    """
    if ":" not in spec:
        return (spec.strip(), None)

    colon_idx = spec.index(":")
    field = spec[:colon_idx].strip()
    rest = spec[colon_idx + 1 :]

    # Split rest on ":" that are outside parentheses
    segments = []
    depth = 0
    current = ""
    for ch in rest:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == ":" and depth == 0:
            segments.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        segments.append(current.strip())

    def _coerce(v):
        v = v.strip()
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    value_groups = []
    for seg in segments:
        eq_idx = seg.index("=")
        vals_str = seg[:eq_idx].strip()
        label = seg[eq_idx + 1 :].strip()
        if vals_str.startswith("(") and vals_str.endswith(")"):
            vals_str = vals_str[1:-1]
        vals = [_coerce(v) for v in vals_str.split(",")]
        value_groups.append((vals, label))

    return (field, value_groups)


def auto_group(models, group_by_specs):
    """Derive groups from models based on field value combinations.

    group_by_specs: list of (field, value_groups) tuples where:
      - field: a MODEL_NAMES field name
      - value_groups: None to derive groups from unique values, or a list of
        ([val, ...], label) to map specific values to a custom label.

    Returns [(label, [model_keys]), ...] ordered by first appearance.
    """
    group_keys = {}
    group_models = {}

    for model_key in models:
        info = MODEL_NAMES.get(model_key, {})
        key_parts = []
        label_parts = []

        for field, value_groups in group_by_specs:
            val = info.get(field)
            if value_groups is not None:
                # Map the model's value to whichever specified group contains it
                matched_key = val  # fallback: raw value (unmatched)
                matched_label = (
                    _short_param_value(field, val) if val is not None else None
                )
                for group_vals, group_label in value_groups:
                    if val in group_vals:
                        matched_key = group_label
                        matched_label = group_label
                        break
                key_parts.append(matched_key)
                if matched_label is not None:
                    label_parts.append(matched_label)
            else:
                key_parts.append(val)
                if val is not None:
                    label_parts.append(_short_param_value(field, val))

        key = tuple(key_parts)
        if key not in group_keys:
            group_keys[key] = ", ".join(label_parts) if label_parts else repr(key)
            group_models[key] = []
        group_models[key].append(model_key)

    return [(group_keys[k], group_models[k]) for k in group_keys]


def build_table(
    models,
    scenes,
    results_dir,
    metrics=None,
    font_size="small",
    smaller_models=0,
    smaller_metrics=0,
    groups=None,
    model_params=None,
    group_params=None,
    rotate_models=False,
    metric_vlines=False,
    value_sep=None,
    metric_sep=None,
    cell_highlights=None,
    group_left_label="",
    model_name_overrides=None,
):
    """Build a LaTeX tabular string.

    Column layout is always metrics-outer → (groups-inner →) models-innermost,
    matching the current behaviour when no groups are provided.

    groups: optional list of (label, [model_keys]) tuples. When provided a group
            header row is inserted between the metric and model header rows,
            repeating the group labels under each metric block. When None all
            models form a single anonymous group with no group header row.
    model_params: MODEL_NAMES field names to annotate individual model column headers.
    group_params: MODEL_NAMES field names whose unique values per group are appended
                  to each group label (e.g. ["splats"] → "Random (10000, 9999 Splats)").
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    if groups is not None:
        resolved_groups = groups
        show_group_header = True
    else:
        resolved_groups = [("", models)]
        show_group_header = False

    if cell_highlights is None:
        cell_highlights = []
    else:
        cell_highlights = list(map(get_cell_highlight, cell_highlights))

    all_models = [m for _, gm in resolved_groups for m in gm]

    font_size_index = get_font_size_index(font_size)
    font_size = LATEX_FONT_SIZES[font_size_index]
    model_font_size = LATEX_FONT_SIZES[max(0, font_size_index - smaller_models)]
    metric_font_size = LATEX_FONT_SIZES[max(0, font_size_index - smaller_metrics)]

    # Load all data up front
    val_data = {m: {} for m in all_models}
    mem_data = {m: {} for m in all_models}
    ckpt_sizes = {m: {} for m in all_models}
    needs_val = any(METRIC_DEFS[m]["source"] == "val" for m in metrics)
    needs_mem = any(METRIC_DEFS[m]["source"] == "mem" for m in metrics)
    needs_ckpt = any(METRIC_DEFS[m]["source"] == "ckpt" for m in metrics)

    for model in all_models:
        for scene in scenes:
            if scene not in COMMAND_SCENES and not scene.startswith("rename_to_"):
                if needs_val:
                    val_data[model][scene] = last_val_stats(results_dir, model, scene)
                if needs_mem:
                    mem_data[model][scene] = last_mem_val_stats(
                        results_dir, model, scene
                    )
                if needs_ckpt:
                    ckpt_sizes[model][scene] = last_ckpt_size(results_dir, model, scene)

    n_metrics = len(metrics)
    # Number of model columns per metric block (same for every metric)
    n_models_per_block = sum(len(gm) for _, gm in resolved_groups)

    metric_block = "c" * n_models_per_block
    sep = rf"@{{\hspace{{{metric_sep}}}}}" if metric_sep else ""
    if metric_vlines:
        # sep goes after each | so there is breathing room between the rule and the columns
        col_spec = "|l|" + "|".join(sep + metric_block for _ in metrics) + "|"
    else:
        # sep between metric blocks only (not before the first)
        col_spec = "l" + metric_block + (sep + metric_block) * (n_metrics - 1)
    lines = []
    lines.append(rf"\{font_size}")
    if value_sep is not None:
        lines.append(rf"\setlength{{\tabcolsep}}{{{value_sep}}}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Cmidrules spanning each metric's full column block
    def metric_span_cmidrules():
        rules = r"\cmidrule{1-1}"
        col = 2
        for _ in metrics:
            rules += rf"\cmidrule(lr){{{col}-{col + n_models_per_block - 1}}}"
            col += n_models_per_block
        return rules

    # With metric_vlines, multicolumn cells must carry the | in their own format
    # spec because \multicolumn overrides the column spec for spanned columns.
    scene_fmt = (
        "|c|" if metric_vlines else "c"
    )  # scene col gets left+right | to match table border
    metric_fmt = "c|" if metric_vlines else "c"  # each metric block ends with |

    # --- Metric header row ---
    metric_header = rf"\multicolumn{{1}}{{{scene_fmt}}}{{Metric}}"
    for mkey in metrics:
        metric_header += rf" & \multicolumn{{{n_models_per_block}}}{{{metric_fmt}}}"
        metric_header += "{"
        if smaller_metrics:
            metric_header += rf"\{metric_font_size} "
        metric_header += (
            METRIC_DEFS[mkey]["label"]
            + (r"$\uparrow$" if METRIC_DEFS[mkey]["up_better"] else r"$\downarrow$")
            + "}"
        )
    lines.append(metric_header + r" \\")
    lines.append(metric_span_cmidrules())

    if show_group_header:
        # --- Group header row (group labels repeat under each metric block) ---
        group_header = rf"\multicolumn{{1}}{{{scene_fmt}}}{{{group_left_label}}}"
        n_groups = len(resolved_groups)
        for _ in metrics:
            for g_idx, (label, group_models) in enumerate(resolved_groups):
                n_g = len(group_models)
                # Last group in each metric block gets the metric boundary |
                g_fmt = metric_fmt if (g_idx == n_groups - 1) else "c"
                display_label = get_group_display_name(
                    label, group_models, group_params
                )
                group_header += (
                    rf" & \multicolumn{{{n_g}}}{{{g_fmt}}}{{{display_label}}}"
                )
        lines.append(group_header + r" \\")

        # Cmidrules spanning each group within each metric block
        group_cmidrules = r"\cmidrule{1-1}"
        col = 2
        for _ in metrics:
            for _, group_models in resolved_groups:
                n_g = len(group_models)
                group_cmidrules += rf"\cmidrule(lr){{{col}-{col + n_g - 1}}}"
                col += n_g
        lines.append(group_cmidrules)

    # --- Model header row ---
    # Individual model cells are not multicolumn, so the column spec's | applies
    # automatically. Only the scene cell (which uses multicolumn) needs explicit |.
    model_header = rf"\multicolumn{{1}}{{{scene_fmt}}}{{Scene $|$ Model}}"
    for _ in metrics:
        for _, group_models in resolved_groups:
            for model in group_models:
                display = get_model_display_name(
                    model, model_params, model_name_overrides
                )
                if smaller_models:
                    display = rf"{{\{model_font_size} {display}}}"
                if rotate_models:
                    display = rf"\rotatebox{{270}}{{{display}}}"
                model_header += f" & {display}"
    lines.append(model_header + r" \\")
    lines.append(r"\midrule")

    def empty_agg():
        return {m: {mkey: [] for mkey in metrics} for m in all_models}

    agg = empty_agg()
    total_agg = empty_agg()

    def highlighted_row(cell_highlights, row_values, mkey):
        row_ranking = list(
            sorted(
                map(
                    swap,
                    filter(circ(issome, proj(1)), enumerate(row_values)),
                ),
                reverse=METRIC_DEFS[mkey]["up_better"],
            )
        )

        row_values = list(map(optfold("--", METRIC_DEFS[mkey]["fmt"]), row_values))

        rank = 0
        i = 0
        while i < len(row_ranking) and rank < len(cell_highlights):
            index = row_ranking[i][1]
            row_values[index] = cell_highlights[rank](row_values[index])
            if i + 1 >= len(row_ranking) or row_ranking[i][0] != row_ranking[i + 1][0]:
                rank = i + 1
            i += 1
        if len(row_values) == 0:
            return ""
        return "&" + "&".join(row_values)

    def avg_row(label, the_agg, cell_highlights):
        row = label
        for mkey in metrics:
            row_values = []

            for _, group_models in resolved_groups:
                for model in group_models:
                    vals = the_agg[model][mkey]
                    row_values.append(ave(vals))
            row += highlighted_row(cell_highlights, row_values, mkey)
        return row + r" \\"

    rename_next = None

    for scene in scenes:
        if scene == "midline":
            lines.append(r"\midrule")
        elif scene == "average":
            name = rename_next or "Average"
            rename_next = None
            lines.append(avg_row(name, agg, cell_highlights))
            agg = empty_agg()
        elif scene == "total_average":
            name = rename_next or "Average"
            rename_next = None
            lines.append(avg_row(name, total_agg, cell_highlights))
        elif scene.startswith("rename_to_"):
            rename_next = scene[len("rename_to_") :]
        elif scene == "remove_prev":
            lines.pop()
        else:
            name = rename_next or r"\ "[0] + scene.capitalize().replace("_", r"\_")
            rename_next = None
            row = name
            for mkey in metrics:
                row_values = []
                for _, group_models in resolved_groups:
                    for model in group_models:
                        vd = val_data[model].get(scene)
                        md = mem_data[model].get(scene)
                        cs = ckpt_sizes[model].get(scene)
                        v = _get_metric_value(mkey, vd, md, cs)
                        if v is None:
                            row_values.append(None)
                        else:
                            row_values.append(v)
                            agg[model][mkey].append(v)
                            total_agg[model][mkey].append(v)

                row += highlighted_row(cell_highlights, row_values, mkey)
            lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


DATASETS = {
    "mip_nerf_360": [
        "bonsai",
        "counter",
        "kitchen",
        "room",
        "midline",
        "rename_to_Indoor",
        "average",
        "midline",
        "bicycle",
        "flowers",
        "garden",
        "stump",
        "treehill",
        "midline",
        "rename_to_Outdoor",
        "average",
        "midline",
        "rename_to_Overall",
        "total_average",
    ],
    "nerf_synthetic": [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
        "midline",
        "rename_to_Overall",
        "total_average",
    ],
    "mip_nerf_360_ave": [
        "bonsai",
        "remove_prev",
        "counter",
        "remove_prev",
        "kitchen",
        "remove_prev",
        "room",
        "remove_prev",
        "rename_to_Indoor",
        "average",
        "bicycle",
        "remove_prev",
        "flowers",
        "remove_prev",
        "garden",
        "remove_prev",
        "stump",
        "remove_prev",
        "treehill",
        "remove_prev",
        "rename_to_Outdoor",
        "average",
        "midline",
        "rename_to_Overall",
        "total_average",
    ],
    "nerf_synthetic_ave": [
        "chair",
        "remove_prev",
        "drums",
        "remove_prev",
        "ficus",
        "remove_prev",
        "hotdog",
        "remove_prev",
        "lego",
        "remove_prev",
        "materials",
        "remove_prev",
        "mic",
        "remove_prev",
        "ship",
        "remove_prev",
        "rename_to_Overall",
        "total_average",
    ],
}

DATASETS["mn360"] = DATASETS["mip_nerf_360"]
DATASETS["ns"] = DATASETS["nerf_synthetic"]
DATASETS["mn360a"] = DATASETS["mip_nerf_360_ave"]
DATASETS["nsa"] = DATASETS["nerf_synthetic_ave"]


def _parse_group(s):
    """Parse 'Label:model1,model2' into ('Label', ['model1', 'model2'])."""
    label, sep, model_str = s.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"Group must be in 'Label:model1,model2' format, got: {s!r}"
        )
    model_keys = [m.strip() for m in model_str.split(",") if m.strip()]
    if not model_keys:
        raise argparse.ArgumentTypeError(f"Group {label!r} has no models: {s!r}")
    return (label, model_keys)


def main():
    parser = argparse.ArgumentParser(
        description="Compile a LaTeX table from eval results"
    )

    # Model specification: either a flat list or explicit named groups
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help=(
            "Model directory names under results/. "
            "Required when not using explicit --groups, and also when using "
            "--groups with value-grouping syntax."
        ),
    )
    parser.add_argument(
        "--groups",
        "-g",
        nargs="+",
        metavar="SPEC",
        help=(
            "Named model groups. Two formats are accepted: "
            "(1) Explicit: 'Label:model1,model2' — lists models directly; "
            "(2) Value-grouping: 'field:(v1,v2)=label:(v3,v4)=label2' — assigns "
            "models (from --models) to groups based on their parameter values. "
            "Example: --groups 'splats:(2000,1966)=~2k:(10000,9833)=~10k'"
        ),
    )

    parser.add_argument(
        "--group-by",
        "-gb",
        nargs="+",
        choices=ANNOTATABLE_PARAMS,
        default=None,
        metavar="PARAM",
        help=(
            "Auto-derive group headers by grouping --models on unique combinations "
            "of the given MODEL_NAMES field(s). Groups are ordered by first "
            "appearance in --models. "
            "Example: --group-by init  groups models into 'Random' and 'SfM' buckets."
        ),
    )

    parser.add_argument("--scenes", "-s", nargs="+", help="Scene names")
    parser.add_argument("--datasets", "-ds", nargs="+", help="Dataset names")
    parser.add_argument(
        "--metrics",
        "-mt",
        nargs="+",
        default=DEFAULT_METRICS,
        choices=list(METRIC_DEFS.keys()),
        help=(
            f"Metrics to include (default: {' '.join(DEFAULT_METRICS)}). "
            f"Available: {', '.join(METRIC_DEFS.keys())}"
        ),
    )
    parser.add_argument(
        "--cell-highlights",
        "-ch",
        nargs="+",
        default=None,
        help=("How cells should be highlighted (in order of rank). "),
    )
    parser.add_argument(
        "--model-params",
        "-mp",
        nargs="+",
        choices=ANNOTATABLE_PARAMS,
        default=None,
        metavar="PARAM",
        help=(
            "MODEL_NAMES fields to annotate each model's column header with. "
            r"'tex_grad' appends '$\partial$' when True; "
            "'splats' and 'init' appear in parentheses. "
            r"Example: --model-params tex_grad splats  →  'TGS$\partial$ (10000 Splats)'"
        ),
    )
    parser.add_argument(
        "--model-names",
        "-mn",
        nargs="+",
        default=None,
        metavar="KEY=NAME",
        help=(
            "Override display names for specific models. "
            "Format: model_key=Display Name (spaces allowed in the name). "
            "Example: --model-names tgs=TGS tgs_b2='TGS w/ TexGrad'"
        ),
    )
    parser.add_argument(
        "--group-params",
        "-gp",
        nargs="+",
        choices=ANNOTATABLE_PARAMS,
        default=None,
        metavar="PARAM",
        help=(
            "MODEL_NAMES fields whose unique values across each group's models are "
            "appended to the group label in parentheses. "
            "Example: --group-params splats  →  'Random (10000 Splats, 9999 Splats)'"
        ),
    )
    parser.add_argument(
        "--results_dir",
        "-rd",
        default=RESULTS_DIR,
        help="Path to results directory (default: ../results relative to this script)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output .tex file path (default: print to stdout)",
    )
    parser.add_argument(
        "--font-size", "-fs", default="small", help="The latex font size of the table"
    )
    parser.add_argument(
        "--smaller-models",
        "-sml",
        nargs="?",
        const=1,
        default=0,
        type=int,
        metavar="N",
        help="Reduce model header font size by N levels (default 1 when flag is given)",
    )
    parser.add_argument(
        "--smaller-metrics",
        "-smt",
        nargs="?",
        const=1,
        default=0,
        type=int,
        metavar="N",
        help="Reduce metric header font size by N levels (default 1 when flag is given)",
    )
    parser.add_argument(
        "--metric-vlines",
        "-mvl",
        action="store_true",
        help="Add vertical lines between metric groups in the column spec",
    )
    parser.add_argument(
        "--value-sep",
        "-vs",
        default=None,
        metavar="LENGTH",
        help=(
            "Horizontal space between values (base column separation), as a LaTeX length. "
            "Example: --value-sep 6pt  or  -value-sep 1em"
        ),
    )
    parser.add_argument(
        "--metric-sep",
        "-ms",
        default=None,
        metavar="LENGTH",
        help=(
            "Extra horizontal space between metric groups, as a LaTeX length. "
            "Example: --metric-sep 6pt  or  --metric-sep 1em"
        ),
    )
    parser.add_argument(
        "--rotate-models",
        "-rm",
        action="store_true",
        help="Rotate model header names 90° using \\rotatebox{90}{...} (requires graphicx package)",
    )
    args = parser.parse_args()

    if args.models is None and args.groups is None:
        parser.error("one of --models/-m or --groups/-g is required")

    if args.groups is not None:
        if args.group_by is not None:
            parser.error("--group-by cannot be used with --groups")
        if any("=" in g for g in args.groups):
            # Extended value-grouping format: "field:(v1,v2)=label:(v3,v4)=label2"
            if not args.models:
                parser.error("--groups with value-grouping syntax requires --models")
            models = args.models
            group_by_specs = [_parse_group_by_spec(s) for s in args.groups]
            groups = auto_group(models, group_by_specs)
            group_left_label = ", ".join(
                PARAM_LABELS.get(field, field.replace("_", " ").title())
                for field, _ in group_by_specs
            )
        else:
            # Explicit format: "Label:model1,model2"
            parsed_groups = [_parse_group(g) for g in args.groups]
            models = [m for _, gm in parsed_groups for m in gm]
            groups = parsed_groups
            group_left_label = ""
    else:
        models = args.models
        if args.group_by:
            group_by_specs = [_parse_group_by_spec(s) for s in args.group_by]
            groups = auto_group(models, group_by_specs)
            group_left_label = ", ".join(
                PARAM_LABELS.get(field, field.replace("_", " ").title())
                for field, _ in group_by_specs
            )
        else:
            groups = None
            group_left_label = ""

    model_name_overrides = None
    if args.model_names:
        model_name_overrides = {}
        for item in args.model_names:
            key, sep, name = item.partition("=")
            if not sep:
                parser.error(f"--model-names entries must be KEY=NAME, got: {item!r}")
            model_name_overrides[key.strip()] = name.strip()

    scenes = args.scenes
    if not scenes:
        scenes = []
        for dataset in args.datasets:
            scenes += DATASETS[dataset]

    table = build_table(
        models,
        scenes,
        args.results_dir,
        metrics=args.metrics,
        font_size=args.font_size,
        smaller_models=args.smaller_models,
        smaller_metrics=args.smaller_metrics,
        groups=groups,
        model_params=args.model_params,
        group_params=args.group_params,
        rotate_models=args.rotate_models,
        metric_vlines=args.metric_vlines,
        metric_sep=args.metric_sep,
        value_sep=args.value_sep,
        cell_highlights=args.cell_highlights,
        group_left_label=group_left_label,
        model_name_overrides=model_name_overrides,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(table + "\n")
        print(f"Written to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
