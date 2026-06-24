#!/usr/bin/env python3
"""
Plot per-scene or grouped-average metric differences between comparison models
and a baseline.

Modes
-----
  --mode scenes  (default)
      X-axis = individual scene names. One marker series per model.

  --mode groups
      X-axis = group labels, each plotted as the mean diff over its scenes.
      One marker per model per group.

      Groups are derived automatically from dataset command markers
      (rename_to_X / average / total_average) when --datasets is given, e.g.:
        --datasets mn360   →  Indoor, Outdoor, Overall
        --datasets ns      →  Overall
      Or specify explicitly:
        --groups "Indoor:bonsai,counter,kitchen,room" "Outdoor:bicycle,garden,..."

  In both modes, connect points with lines using --connect.

Axis ranges
-----------
  (default)              matplotlib auto
  --axis-range metric    per-metric "reasonable" difference bounds
  --ylim ymin ymax       same hardcoded bounds for all subplots

For up_better metrics (PSNR, SSIM, CVVDP) the y-axis is normal.
For down_better metrics (LPIPS, time, memory) the y-axis is inverted so the
visually highest position is always the best result.
"""

import argparse, sys, os

sys.path.insert(0, os.path.dirname(__file__))

from compile_latex_table import (
    last_val_stats,
    last_mem_val_stats,
    last_ckpt_size,
    METRIC_DEFS,
    DEFAULT_METRICS,
    DATASETS,
    RESULTS_DIR,
    COMMAND_SCENES,
    get_model_display_name,
    _get_metric_value,
    ANNOTATABLE_PARAMS,
)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

# Reasonable y-ranges used when --axis-range metric is given.
# Data coordinates; y-inversion for down_better metrics is applied separately.
METRIC_DIFF_RANGES = {
    "psnr": (-3.0, 3.0),  # dB
    "ssim": (-0.05, 0.05),
    "lpips": (-0.05, 0.05),
    "cvvdp": (-0.5, 0.5),  # JOD
    "render_time": (-0.02, 0.02),  # seconds
    "memory": (-2.0, 2.0),  # GB
    "model_size": (-5e7, 5e7),  # bytes
}

METRIC_RAW_RANGES = {
    "psnr": (20.0, 40.0),  # dB
    "ssim": (0.5, 1.0),
    "lpips": (0.0, 0.5),
    "cvvdp": (5.0, 10.0),  # JOD
    "render_time": (0.0, 0.1),  # seconds
    "memory": (0.0, 20.0),  # GB
    "model_size": (0.0, 5e8),  # bytes
}

MARKERS = ["x", "+", "o", "s", "^", "D", "v", "P", "*", "h"]
# Markers with no fill area — colour is set via 'color', not 'edgecolors'
_FACELESS_MARKERS = {"x", "+", "|", "_"}


def get_val(mkey, model, scene, val_data, mem_data, ckpt_data):
    return _get_metric_value(
        mkey,
        val_data[model].get(scene),
        mem_data[model].get(scene),
        ckpt_data[model].get(scene),
    )


def load_all(all_models, scenes, metrics, results_dir):
    val_data = {m: {} for m in all_models}
    mem_data = {m: {} for m in all_models}
    ckpt_data = {m: {} for m in all_models}
    needs_val = any(METRIC_DEFS[k]["source"] == "val" for k in metrics)
    needs_mem = any(METRIC_DEFS[k]["source"] == "mem" for k in metrics)
    needs_ckpt = any(METRIC_DEFS[k]["source"] == "ckpt" for k in metrics)
    for model in all_models:
        for scene in scenes:
            if needs_val:
                val_data[model][scene] = last_val_stats(results_dir, model, scene)
            if needs_mem:
                mem_data[model][scene] = last_mem_val_stats(results_dir, model, scene)
            if needs_ckpt:
                ckpt_data[model][scene] = last_ckpt_size(results_dir, model, scene)
    return val_data, mem_data, ckpt_data


def parse_groups_from_raw_scenes(raw_scenes):
    """Derive (label, [scene_names]) groups from a raw scene list that may
    contain command markers, mirroring the aggregation in compile_latex_table.py.

      average       → emit a group from scenes since the last average, then reset
      total_average → emit a group from ALL non-command scenes seen so far
      rename_to_X   → label for the next average / total_average
      remove_prev   → undo the last scene (used in *_ave dataset variants)
      midline       → ignored
    """
    groups = []
    current = []  # scenes in current sub-group (reset after each 'average')
    total = []  # all scenes accumulated (for total_average)
    pending_label = None

    for s in raw_scenes:
        if s == "midline":
            continue
        elif s.startswith("rename_to_"):
            pending_label = s[len("rename_to_") :]
        elif s == "average":
            label = pending_label or "Average"
            pending_label = None
            if current:
                groups.append((label, list(current)))
            current = []
        elif s == "total_average":
            label = pending_label or "Overall"
            pending_label = None
            if total:
                groups.append((label, list(total)))
        elif s == "remove_prev":
            if current:
                current.pop()
            if total:
                total.pop()
        else:
            current.append(s)
            total.append(s)

    # If no average/total_average markers, treat all scenes as one group
    if not groups and total:
        groups.append(("Average", list(total)))

    return groups


def parse_explicit_groups(specs):
    """Parse 'Label:s1,s2,...' strings into (label, [scenes]) tuples.
    Scene names are lowercased to match results directory conventions."""
    groups = []
    for spec in specs:
        label, sep, scene_str = spec.partition(":")
        if not sep:
            raise ValueError(
                f"--groups entries must be 'Label:s1,s2,...', got: {spec!r}"
            )
        scenes = [s.strip().lower() for s in scene_str.split(",") if s.strip()]
        groups.append((label.strip(), scenes))
    return groups


def parse_model_overrides(specs, scene_groups=None):
    """Parse 'model:s1,s2=override' specs into a {(model, scene): override} dict.

    Each part before the '=' can be:
      - a scene name           e.g. 'bonsai'
      - a dataset alias        e.g. 'ns', 'mn360'  (expanded to all real scenes)
      - a group label          e.g. 'Indoor'        (expanded to that group's scenes)
    Multiple items can be comma-separated, e.g. 'tgs:ns,Indoor=tgs_g2000'.
    """
    _skip = COMMAND_SCENES | {"remove_prev"}

    def dataset_scenes(name):
        return [
            s for s in DATASETS[name]
            if s not in _skip and not s.startswith("rename_to_")
        ]

    group_label_to_scenes = {}
    if scene_groups:
        for label, scenes in scene_groups:
            group_label_to_scenes[label] = scenes

    overrides = {}
    for spec in specs:
        model_part, _, rest = spec.partition(":")
        if not rest or "=" not in rest:
            raise ValueError(
                f"--model-overrides entries must be 'model:s1,s2=override', got: {spec!r}"
            )
        scenes_part, _, override_key = rest.partition("=")
        model_key    = model_part.strip()
        override_key = override_key.strip()

        resolved = []
        for part in (p.strip() for p in scenes_part.split(",")):
            if part in DATASETS:
                resolved.extend(dataset_scenes(part))
            elif part in group_label_to_scenes:
                resolved.extend(group_label_to_scenes[part])
            else:
                resolved.append(part.lower())

        for scene in resolved:
            overrides[(model_key, scene)] = override_key

    return overrides


def apply_ylim(ax, mkey, axis_range, ylim_fixed, diff_mode):
    """Set y-axis data limits then invert for down_better metrics."""
    if ylim_fixed is not None:
        ax.set_ylim(*ylim_fixed)
    elif axis_range == "metric":
        ranges = METRIC_DIFF_RANGES if diff_mode else METRIC_RAW_RANGES
        ax.set_ylim(*ranges[mkey])
    if not METRIC_DEFS[mkey]["up_better"]:
        ax.invert_yaxis()


def main():
    parser = argparse.ArgumentParser(
        description="Plot metric differences between comparison models and a baseline"
    )
    parser.add_argument(
        "--baseline",
        "-b",
        default=None,
        help="Baseline model key. If omitted, raw metric values are plotted "
        "instead of differences.",
    )
    parser.add_argument(
        "--models", "-m", nargs="+", required=True, help="Comparison model keys"
    )
    parser.add_argument(
        "--datasets", "-ds", nargs="+", help="Dataset aliases (ns, mn360, …)"
    )
    parser.add_argument(
        "--scenes",
        "-s",
        nargs="+",
        help="Explicit flat scene list (overrides --datasets)",
    )
    parser.add_argument(
        "--metrics",
        "-mt",
        nargs="+",
        default=DEFAULT_METRICS,
        choices=list(METRIC_DEFS.keys()),
    )
    parser.add_argument(
        "--mode",
        choices=["scenes", "groups"],
        default="scenes",
        help=(
            "scenes: one x-tick per scene, marker per model; "
            "groups: one x-tick per named group of scenes, "
            "each point is the mean diff over that group's scenes"
        ),
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        metavar="LABEL:s1,s2,...",
        help=(
            "Explicit scene groups for --mode groups. "
            "If omitted with --datasets, groups are derived "
            "automatically from the dataset's average / total_average markers."
        ),
    )
    parser.add_argument(
        "--model-overrides",
        nargs="+",
        default=None,
        metavar="MODEL:SCENES=OVERRIDE",
        help=(
            "Per-scene model substitutions. Format: 'model:s1,s2=override_model'. "
            "In --mode groups, group labels can be used instead of scene names, "
            "e.g. 'tgs:Indoor=tgs_psfm'. Applied to both baseline and comparison "
            "models; specify 'baseline' as the model key to override the baseline."
        ),
    )
    parser.add_argument(
        "--connect",
        action="store_true",
        help="Connect points with lines (off by default in both modes)",
    )
    parser.add_argument(
        "--axis-range",
        choices=["auto", "metric"],
        default="auto",
        help=(
            "auto: matplotlib default ranges; "
            "metric: per-metric reasonable difference bounds"
        ),
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Hardcoded y-axis data limits for all subplots "
        "(overrides --axis-range; axis is still inverted "
        "for down_better metrics within these limits)",
    )
    parser.add_argument("--results_dir", "-rd", default=RESULTS_DIR)
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output image path (default: show interactively)",
    )
    parser.add_argument(
        "--model-names", "-mn", nargs="+", default=None, metavar="KEY=NAME"
    )
    parser.add_argument(
        "--model-params", "-mp", nargs="+", default=None, choices=ANNOTATABLE_PARAMS
    )
    parser.add_argument(
        "--baseline-name",
        default=None,
        help="Display name for the baseline in the legend title",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=None,
        help="Width per subplot in inches (default: auto)",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=3.5,
        help="Figure height in inches (default: 3.5)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve raw scene list                                               #
    # ------------------------------------------------------------------ #
    if args.scenes:
        raw_scenes = args.scenes
    elif args.datasets:
        raw_scenes = []
        for ds in args.datasets:
            raw_scenes += DATASETS[ds]
    else:
        parser.error("one of --scenes or --datasets is required")

    # ------------------------------------------------------------------ #
    # Determine x-axis structure                                           #
    # ------------------------------------------------------------------ #
    if args.mode == "groups":
        if args.groups:
            scene_groups = parse_explicit_groups(args.groups)
        else:
            scene_groups = parse_groups_from_raw_scenes(raw_scenes)
        if not scene_groups:
            parser.error(
                "No scene groups could be derived. "
                "Use --groups or a dataset with average/total_average markers."
            )
        x_labels = [label for label, _ in scene_groups]
        all_plot_scenes = list(dict.fromkeys(s for _, gs in scene_groups for s in gs))
    else:  # scenes
        skip = COMMAND_SCENES | {"remove_prev"}
        all_plot_scenes = [
            s for s in raw_scenes if s not in skip and not s.startswith("rename_to_")
        ]
        x_labels = [s.replace("_", " ").capitalize() for s in all_plot_scenes]
        scene_groups = None

    # ------------------------------------------------------------------ #
    # Model name overrides                                                 #
    # ------------------------------------------------------------------ #
    model_name_overrides = {}
    if args.model_names:
        for item in args.model_names:
            key, sep, name = item.partition("=")
            if not sep:
                parser.error(f"--model-names entries must be KEY=NAME, got: {item!r}")
            model_name_overrides[key.strip()] = name.strip()

    diff_mode = args.baseline is not None
    if diff_mode:
        baseline_display = (
            args.baseline_name
            or model_name_overrides.get(args.baseline)
            or get_model_display_name(
                args.baseline, args.model_params, model_name_overrides
            )
        )
    else:
        baseline_display = None

    def model_label(model):
        return get_model_display_name(model, args.model_params, model_name_overrides)

    # ------------------------------------------------------------------ #
    # Model overrides (per-scene substitutions)                            #
    # ------------------------------------------------------------------ #
    model_overrides = {}
    if args.model_overrides:
        model_overrides = parse_model_overrides(args.model_overrides, scene_groups)

    def resolve_model(model, scene):
        """Return the actual model key to load results from for this (model, scene) pair."""
        return model_overrides.get((model, scene), model)

    # ------------------------------------------------------------------ #
    # Load data                                                            #
    # ------------------------------------------------------------------ #
    # Include override model keys alongside the declared models
    override_keys = list(dict.fromkeys(model_overrides.values()))
    all_models = list(
        dict.fromkeys(
            ([args.baseline] if diff_mode else []) + args.models + override_keys
        )
    )
    metrics = args.metrics
    val_data, mem_data, ckpt_data = load_all(
        all_models, all_plot_scenes, metrics, args.results_dir
    )

    # ------------------------------------------------------------------ #
    # Compute y-values                                                     #
    # plot_vals[m_idx][metric_idx] = array of y-values (one per x-tick)   #
    # ------------------------------------------------------------------ #
    plot_vals = []
    for model in args.models:
        model_vals = []
        for mkey in metrics:
            if args.mode == "scenes":
                ys = []
                for scene in all_plot_scenes:
                    vc = get_val(
                        mkey,
                        resolve_model(model, scene),
                        scene,
                        val_data,
                        mem_data,
                        ckpt_data,
                    )
                    if diff_mode:
                        vb = get_val(
                            mkey,
                            resolve_model(args.baseline, scene),
                            scene,
                            val_data,
                            mem_data,
                            ckpt_data,
                        )
                        ys.append(
                            vc - vb if (vb is not None and vc is not None) else np.nan
                        )
                    else:
                        ys.append(vc if vc is not None else np.nan)
            else:  # groups
                ys = []
                for _, group_scenes in scene_groups:
                    vals = []
                    for scene in group_scenes:
                        vc = get_val(
                            mkey,
                            resolve_model(model, scene),
                            scene,
                            val_data,
                            mem_data,
                            ckpt_data,
                        )
                        if diff_mode:
                            vb = get_val(
                                mkey,
                                resolve_model(args.baseline, scene),
                                scene,
                                val_data,
                                mem_data,
                                ckpt_data,
                            )
                            if vb is not None and vc is not None:
                                vals.append(vc - vb)
                        else:
                            if vc is not None:
                                vals.append(vc)
                    ys.append(np.mean(vals) if vals else np.nan)
            model_vals.append(ys)
        plot_vals.append(model_vals)

    # ------------------------------------------------------------------ #
    # Figure                                                               #
    # ------------------------------------------------------------------ #
    n_metrics = len(metrics)
    n_models = len(args.models)
    n_xticks = len(x_labels)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_models))

    subplot_w = args.fig_width or max(3.5, n_xticks * 0.75 + 1.5)
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(subplot_w * n_metrics, args.fig_height),
        squeeze=False,
    )
    axes = axes[0]
    x = np.arange(n_xticks)

    for ax_idx, (ax, mkey) in enumerate(zip(axes, metrics)):
        defn = METRIC_DEFS[mkey]

        for m_idx, model in enumerate(args.models):
            ys     = plot_vals[m_idx][ax_idx]
            label  = model_label(model)
            marker   = MARKERS[m_idx % len(MARKERS)]
            color    = colors[m_idx]
            faceless = marker in _FACELESS_MARKERS
            if args.connect:
                ax.plot(
                    x, ys,
                    linewidth=1.0, markersize=6,
                    marker=marker, color=color,
                    markerfacecolor=color if faceless else "none",
                    markeredgecolor=color,
                    label=label, zorder=4,
                )
            else:
                if faceless:
                    ax.scatter(x, ys, s=40, marker=marker, color=color,
                               label=label, zorder=4)
                else:
                    ax.scatter(x, ys, s=40, marker=marker,
                               facecolors="none", edgecolors=color,
                               label=label, zorder=4)

        # Vertical range lines: min to max across models per column
        for xi in range(n_xticks):
            col_vals = [
                plot_vals[m_idx][ax_idx][xi]
                for m_idx in range(n_models)
                if not np.isnan(plot_vals[m_idx][ax_idx][xi])
            ]
            if len(col_vals) >= 2:
                ax.vlines(
                    xi,
                    min(col_vals),
                    max(col_vals),
                    color="black",
                    linewidth=1.2,
                    zorder=3,
                )

        if diff_mode:
            ax.axhline(0, color="black", linewidth=0.8, zorder=2)
        ax.set_title(defn["label"], fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="both", linewidth=0.4, alpha=0.5, zorder=1)

        arrow = r"$\uparrow$" if defn["up_better"] else r"$\downarrow$"
        prefix = "Δ " if diff_mode else ""
        ax.set_ylabel(f"{prefix}{defn['label']} ({arrow})", fontsize=7)

        apply_ylim(ax, mkey, args.axis_range, args.ylim, diff_mode)

    handles, labels = axes[0].get_legend_handles_labels()
    legend_kw = dict(
        loc="upper center",
        ncol=n_models,
        fontsize=8,
        title_fontsize=8,
        bbox_to_anchor=(0.5, 1.0),
    )
    if diff_mode:
        legend_kw["title"] = f"vs. {baseline_display}"
    fig.legend(handles, labels, **legend_kw)

    # Reserve space at the top for the legend.
    # 1 row of entries always; title adds a second line in diff mode.
    legend_lines = 1 + int(diff_mode)
    legend_h_in = legend_lines * 0.28 + 0.15  # per-line height + padding, inches
    top_frac = max(0.5, 1.0 - legend_h_in / args.fig_height)
    fig.tight_layout(rect=[0, 0, 1, top_frac])

    if args.output:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
