#!/usr/bin/env python3
"""
Print a histogram of a model parameter aggregated over scenes.

Usage — simple (one histogram per model, scenes shared):
  python scripts/param_histogram.py \
    --param opacities --models 2dgs tgs --datasets ns

  python scripts/param_histogram.py \
    --param scales --dim 1 --models 2dgs tgs --scenes room bonsai

Usage — groups (each group aggregates multiple (model, scene/dataset) pairs):
  python scripts/param_histogram.py \
    --param scale_ratio \
    --groups "2dgs,bonsai,2dgs_g2000,ns=2DGS" \
             "2dgs_oquad1,bonsai,2dgs_g2000_oquad1,ns=2DGS w/ OpacLoss"

  Group format: "model1,scene_or_dataset1,model2,scene_or_dataset2,...=Label"
  Dataset shorthands (ns, mn360, nerf_synthetic, mip_nerf_360) are expanded
  automatically. Each pair (model, scene_or_dataset) contributes its splats.

Weighting:
  --weight splat  (default)  each splat contributes equally; scenes with more
                             splats have proportionally more influence
  --weight scene             each (model, scene) pair contributes equally,
                             regardless of splat count

Available raw params (from splats dict):
  means  opacities  quats  scales  sh0  shN  steepnesses  textures

Composite params (computed from splats):
  opacities_activated      sigmoid(opacities)
  scales_activated        exp(scales[:, dim])
  scale_ratio            exp(max_scale) / exp(min_scale)  per splat (all 3 dims)
  scale_ratio_2d         same, using only dims 0–1  (correct for 2DGS)
  scale_area             product of exp(scales) across dims  per splat (all 3 dims)
  scale_area_2d          same, using only dims 0–1  (correct for 2DGS)
  scale_log_ratio        max_scale - min_scale in log space  per splat (all 3 dims)
  scale_log_ratio_2d     same, using only dims 0–1  (correct for 2DGS)
  scale_effective_rank   exp(entropy of normalised scales)  1=needle, 3=sphere (all 3 dims)
  scale_effective_rank_2d  same, using only dims 0–1  1=needle, 2=disc  (correct for 2DGS)
  steepnesses_activated  softplus(steepnesses) + 1  (2DGSS models)
  means_displacement     L2 distance each splat moved from its pretrained-init position
                         (reads pretrained_path from cfg.yml; TGS transfer models only)
"""

import argparse
import glob
import inspect
import os
import re
import sys
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DATASETS = {
    "mip_nerf_360": [
        "bonsai",
        "counter",
        "kitchen",
        "room",
        "bicycle",
        "flowers",
        "garden",
        "stump",
        "treehill",
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
    ],
}
DATASETS["mn360"] = DATASETS["mip_nerf_360"]
DATASETS["ns"] = DATASETS["nerf_synthetic"]


def _step_number(path):
    m = re.search(r"(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def last_ckpt(results_dir, model, scene):
    pattern = os.path.join(results_dir, model, scene, "ckpts", "ckpt_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=_step_number)


# ---------------------------------------------------------------------------
# Composite parameter definitions
# Each callable receives the splats dict (and optionally dim) and returns a
# 1-D float tensor.
# ---------------------------------------------------------------------------

COL_CHANNEL_NAMES = ["Red", "Green", "Blue"]

PARAMS = {
    "means": {"label": lambda dim: f"Stored Mean[{dim}]"},
    "opacities": {"label": "Stored Opacities"},
    "quats": {"label": lambda dim: f"Stored Quat[{dim}]"},
    "scales": {"label": lambda dim: f"Stored Scale[{dim}]"},
    "sh0": {"label": lambda dim: f"{COL_CHANNEL_NAMES[dim]}"},
    "shN": {"label": lambda dim: f"Sh[{dim}]"},
    "steepnesses": {"label": "Stored Sharpness"},
    "textures": {"label": lambda dim: f"Textures[{dim}]"},
    "opacities_activated": {
        "label": "Opacity",
        "fn": lambda s: torch.sigmoid(s["opacities"]),
    },
    "scales_activated": {
        "label": lambda dim: f"Scale[{dim}]",
        "fn": lambda s, dim=0: torch.exp(s["scales"][:, dim]),
    },
    "scale_ratio": {
        "label": "Scale Ratio",
        "f": lambda s: (
            torch.exp(s["scales"].max(dim=1).values)
            / torch.exp(s["scales"].min(dim=1).values)
        ),
    },
    "scale_ratio_2d": {
        "label": "Scale Ratio",
        "fn": lambda s: (
            torch.exp(s["scales"][:, :2].max(dim=1).values)
            / torch.exp(s["scales"][:, :2].min(dim=1).values)
        ),
    },
    "scale_volume": {
        "label": "Scale Volume",
        "fn": lambda s: torch.exp(s["scales"]).prod(dim=1),
    },
    "scale_area": {
        "label": "Scale Area",
        "fn": lambda s: torch.exp(s["scales"][:, :2]).prod(dim=1),
    },
    "scale_log_ratio": {
        "label": "Log Scale Ratio",
        "fn": lambda s: (s["scales"].max(dim=1).values - s["scales"].min(dim=1).values),
    },
    "scale_log_ratio_2d": {
        "label": "Log Scale Ratio",
        "fn": lambda s: (
            s["scales"][:, :2].max(dim=1).values - s["scales"][:, :2].min(dim=1).values
        ),
    },
    "scale_effective_rank": {
        "label": "Scale Effective Rank",
        "fn": lambda s: _effective_rank(torch.exp(s["scales"])),
    },
    "scale_effective_rank_2d": {
        "label": "Scale Effective Rank",
        "fn": lambda s: _effective_rank(torch.exp(s["scales"][:, :2])),
    },
    "steepnesses_activated": {
        "label": "Steepness",
        "fn": lambda s: torch.nn.functional.softplus(s["steepnesses"]) + 1,
    },
    "means_displacement": {
        "label": "Splat Displacement",
        # context_fn: receives (splats, result_path) instead of just splats
        "context_fn": lambda s, result_path: _means_displacement(s, result_path),
    },
}


def _effective_rank(scales):
    """Roy & Vetterli effective rank: exp(entropy of normalised scale magnitudes).

    Ranges from 1 (needle) to N (sphere/disc for N in-plane axes).
    scales: [N, K] positive tensor (already exponentiated).
    """
    p = scales / scales.sum(dim=1, keepdim=True)
    entropy = -(p * p.log()).sum(dim=1)
    return entropy.exp()


def _read_pretrained_path(cfg_path):
    """Extract pretrained_path from cfg.yml using regex (avoids custom YAML tags)."""
    with open(cfg_path) as f:
        for line in f:
            m = re.match(r"^pretrained_path:\s*(\S+)", line)
            if m:
                return m.group(1)
    return None


def _means_displacement(splats, result_path):
    """Per-splat L2 distance between final means and pretrained-init means.

    Reads cfg.yml from result_path to find pretrained_path, then loads that
    checkpoint. pretrained_path in cfg.yml is relative to the project root
    (parent of results/).
    """
    cfg_path = os.path.join(result_path, "cfg.yml")
    if not os.path.exists(cfg_path):
        raise KeyError(f"cfg.yml not found at {cfg_path}")

    pretrained_rel = _read_pretrained_path(cfg_path)
    if pretrained_rel is None:
        raise KeyError(f"pretrained_path not found in {cfg_path}")

    # pretrained_path is relative to project root (parent of results/)
    results_dir = os.path.dirname(result_path.rstrip("/"))  # strip scene
    results_dir = os.path.dirname(results_dir)  # strip model
    project_root = os.path.dirname(os.path.abspath(results_dir))
    pretrained_abs = os.path.normpath(os.path.join(__file__, "..", pretrained_rel))

    if not os.path.exists(pretrained_abs):
        raise KeyError(f"Pretrained checkpoint not found: {pretrained_abs}")

    init_ckpt = torch.load(pretrained_abs, map_location="cpu", weights_only=True)
    init_means = init_ckpt["splats"]["means"]
    final_means = splats["means"]

    if init_means.shape != final_means.shape:
        raise KeyError(
            f"Splat count mismatch: init {init_means.shape} vs final {final_means.shape}. "
            "Displacement is only defined when splat count is preserved."
        )

    return (final_means - init_means).norm(dim=1)


def extract_values(splats, param, dim, result_path=None):
    if param in PARAMS:
        entry = PARAMS[param]
        if "context_fn" in entry:
            if result_path is None:
                raise KeyError(
                    f"Parameter '{param}' requires a result path (result directory context)."
                )
            vals = entry["context_fn"](splats, result_path)
            return vals.flatten().float().numpy()
        if "fn" in entry:
            fn = entry["fn"]
            sig = inspect.signature(fn)
            vals = fn(splats, dim=dim) if "dim" in sig.parameters else fn(splats)
            return vals.flatten().float().numpy()
        # raw splat tensor
        tensor = splats[param].float()
        if tensor.dim() == 1:
            return tensor.numpy()
        if dim >= tensor.shape[1]:
            raise ValueError(
                f"--dim {dim} out of range for '{param}' with shape {list(tensor.shape)}"
            )
        return tensor[:, dim].flatten().numpy()

    raise KeyError(
        f"Parameter '{param}' not found. "
        f"In checkpoint: {list(splats.keys())} "
        f"Supported: {list(PARAMS.keys())}"
    )


def get_label(param, dim):
    if param in PARAMS:
        if isinstance(PARAMS[param]["label"], Callable):
            return PARAMS[param]["label"](dim)
        else:
            return PARAMS[param]["label"]

    raise KeyError(
        f"Parameter '{param}' not found in labels. " f"Supported: {list(PARAMS.keys())}"
    )


def resolve_scene_or_dataset(token):
    """Return a list of scenes for a scene name or dataset shorthand."""
    return DATASETS.get(token, [token])


def parse_group_spec(spec):
    """Parse "m1,s1,m2,ds2,...=Label" → (label, [(model, [scenes]), ...])."""
    eq_idx = spec.rfind("=")
    if eq_idx == -1:
        raise argparse.ArgumentTypeError(f"Group spec must contain '=': {spec!r}")
    label = spec[eq_idx + 1 :]
    tokens = [t.strip() for t in spec[:eq_idx].split(",")]
    if len(tokens) % 2 != 0:
        raise argparse.ArgumentTypeError(
            f"Group spec must have an even number of comma-separated tokens "
            f"(alternating model, scene_or_dataset): got {tokens!r}"
        )
    model_scenes = []
    for i in range(0, len(tokens), 2):
        model = tokens[i]
        scene_or_ds = tokens[i + 1]
        model_scenes.append((model, resolve_scene_or_dataset(scene_or_ds)))
    return label, model_scenes


def collect_group(results_dir, model_scenes, param, dim, weight_by):
    """
    Collect values for a group of (model, [scenes]) pairs.

    Returns (values, weights) as 1-D numpy arrays, or (None, None) on failure.
    weight_by='splat': uniform weights (each splat counts equally).
    weight_by='scene': weights = 1/N_scene per value so each scene sums to 1.
    """
    all_vals = []
    all_weights = []

    for model, scenes in model_scenes:
        for scene in scenes:
            ckpt_path = last_ckpt(results_dir, model, scene)
            if ckpt_path is None:
                print(f"  [warn] no checkpoint: {model}/{scene}", file=sys.stderr)
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            splats = ckpt.get("splats", {})
            result_path = os.path.join(results_dir, model, scene)
            try:
                vals = extract_values(splats, param, dim, result_path=result_path)
            except KeyError as e:
                print(f"  [skip] {model}/{scene}: {e}", file=sys.stderr)
                return None, None

            n = len(vals)
            w = np.ones(n) / n if weight_by == "scene" else np.ones(n)
            all_vals.append(vals)
            all_weights.append(w)

    if not all_vals:
        return None, None
    return np.concatenate(all_vals), np.concatenate(all_weights)


def _optional_float(value):
    """argparse type that accepts a float or the literal string 'None'."""
    if value.lower() == "none":
        return None
    return float(value)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a histogram of a splat parameter across models/scenes"
    )
    parser.add_argument(
        "--param",
        "-p",
        required=True,
        help=(
            "Parameter name. Raw: means opacities quats scales sh0 shN textures. "
            f"Supported: {' '.join(PARAMS.keys())}"
        ),
    )

    # --- group specification (two mutually exclusive styles) ---
    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument(
        "--models",
        "-m",
        nargs="+",
        help=(
            "One histogram per model; combine with --scenes / --datasets. "
            "Optionally append '=Label' to override the legend name, "
            "e.g. '2dgs=2DGS'."
        ),
    )
    group_mode.add_argument(
        "--groups",
        "-g",
        nargs="+",
        metavar="SPEC",
        help=(
            "Explicit groups. Each SPEC: "
            "'model1,scene_or_ds1,model2,scene_or_ds2,...=Label'. "
            "Dataset shorthands (ns, mn360, …) are expanded automatically."
        ),
    )

    parser.add_argument(
        "--scenes", "-s", nargs="*", default=None, help="Scenes to use with --models"
    )
    parser.add_argument(
        "--datasets",
        "-ds",
        nargs="+",
        default=None,
        help=f"Dataset shorthands to use with --models: {', '.join(DATASETS.keys())}",
    )
    parser.add_argument(
        "--weight",
        "-w",
        choices=["splat", "scene"],
        default="splat",
        help=(
            "Weighting mode. 'splat': each splat counts equally (default). "
            "'scene': each (model, scene) pair counts equally."
        ),
    )
    parser.add_argument(
        "--dim",
        "-d",
        type=int,
        default=0,
        help="Dimension index for multi-dim raw params (default 0). Ignored for composites.",
    )
    parser.add_argument("--bins", "-b", type=int, default=100)
    parser.add_argument(
        "--density",
        action="store_true",
        help="Normalise to a density (integrates to 1).",
    )
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument(
        "--clip",
        nargs=2,
        type=_optional_float,
        metavar=("LO", "HI"),
        default=None,
        help=(
            "Clip values to [LO, HI] before plotting. "
            "Use 'None' to leave one end open (e.g. --clip None 1.0)."
        ),
    )
    parser.add_argument(
        "--percentile",
        nargs=2,
        type=_optional_float,
        metavar=("LO", "HI"),
        default=None,
        help=(
            "Keep only values between the LO-th and HI-th percentile "
            "(computed per group). Use 'None' for an open end "
            "(e.g. --percentile 1 85  or  --percentile None 99)."
        ),
    )
    parser.add_argument("--results-dir", "-rd", default=RESULTS_DIR)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--title", "-t", default=None, help="Override the plot title.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=_optional_float,
        metavar=("LO", "HI"),
        default=None,
        help=(
            "Set x-axis display range without clipping data. "
            "Use 'None' for an open end (e.g. --xlim 0 None)."
        ),
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=_optional_float,
        metavar=("LO", "HI"),
        default=None,
        help=(
            "Set y-axis display range. "
            "Use 'None' for an open end (e.g. --ylim 0 None)."
        ),
    )
    args = parser.parse_args()

    # Build the list of (label, model_scenes) groups
    groups = []  # [(label, [(model, [scenes])])]

    if args.groups is not None:
        for spec in args.groups:
            try:
                label, model_scenes = parse_group_spec(spec)
            except argparse.ArgumentTypeError as e:
                parser.error(str(e))
            groups.append((label, model_scenes))
    else:
        # --models mode: build scenes list from --scenes / --datasets
        scenes = list(args.scenes or [])
        for ds in args.datasets or []:
            if ds not in DATASETS:
                parser.error(
                    f"Unknown dataset '{ds}'. Known: {', '.join(DATASETS.keys())}"
                )
            scenes += DATASETS[ds]
        if not scenes:
            parser.error("Provide at least one scene via --scenes or --datasets")
        for entry in args.models:
            model, _, label = entry.partition("=")
            if not label:
                label = model
            groups.append((label, [(model, scenes)]))

    # --- Pass 1: collect and filter all groups ---
    prepared = []  # [(label, vals, weights)]
    for label, model_scenes in groups:
        vals, weights = collect_group(
            args.results_dir, model_scenes, args.param, args.dim, args.weight
        )
        if vals is None:
            print(f"[skip] '{label}': no data collected", file=sys.stderr)
            continue

        if args.clip is not None:
            lo, hi = args.clip
            mask = np.ones(len(vals), dtype=bool)
            if lo is not None:
                mask &= vals >= lo
            if hi is not None:
                mask &= vals <= hi
            vals, weights = vals[mask], weights[mask]

        finite = np.isfinite(vals)
        if not finite.all():
            print(
                f"  [warn] '{label}': {(~finite).sum()} non-finite values removed",
                file=sys.stderr,
            )
            vals, weights = vals[finite], weights[finite]

        prepared.append((label, vals, weights))

    if not prepared:
        print("No data found for any group.", file=sys.stderr)
        sys.exit(1)

    # --- Compute shared bin edges from the union of all collected values ---
    all_vals_combined = np.concatenate([v for _, v, _ in prepared])

    # Percentile sets the bin range without removing data from any group.
    if args.percentile is not None:
        lo_p, hi_p = args.percentile
        range_lo = float(np.percentile(all_vals_combined, lo_p)) if lo_p is not None else float(all_vals_combined.min())
        range_hi = float(np.percentile(all_vals_combined, hi_p)) if hi_p is not None else float(all_vals_combined.max())
    else:
        range_lo = float(all_vals_combined.min())
        range_hi = float(all_vals_combined.max())

    bin_edges = np.histogram_bin_edges(all_vals_combined, bins=args.bins, range=(range_lo, range_hi))

    # --- Pass 2: plot each group with the shared edges ---
    hist_range = (float(bin_edges[0]), float(bin_edges[-1]))
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, vals, weights in prepared:
        ax.hist(
            vals,
            bins=bin_edges,
            range=hist_range,
            weights=weights,
            density=args.density,
            alpha=args.alpha,
            label=label,
        )

    param_label = get_label(args.param, args.dim)

    if args.density:
        ylabel = "Density"
    elif args.weight == "scene":
        ylabel = "Scene Proportion"
    else:
        ylabel = "Splat Count"

    ax.set_xlabel(param_label)
    ax.set_ylabel(ylabel)
    ax.set_title(
        args.title if args.title is not None else f"Histogram of {param_label}"
    )
    ax.legend()

    if args.log_scale:
        ax.set_yscale("log")

    if args.xlim is not None:
        ax.set_xlim(
            args.xlim[0] if args.xlim[0] is not None else ax.get_xlim()[0],
            args.xlim[1] if args.xlim[1] is not None else ax.get_xlim()[1],
        )
    if args.ylim is not None:
        ax.set_ylim(
            args.ylim[0] if args.ylim[0] is not None else ax.get_ylim()[0],
            args.ylim[1] if args.ylim[1] is not None else ax.get_ylim()[1],
        )

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
