#!/usr/bin/env python3
"""
Compile a LaTeX table from evaluation results.

Usage:
  python scripts/compile_latex_table.py \
    --models 2dgs tgs \
    --scenes chair drums ficus hotdog lego materials mic ship \
    --results_dir results
"""

import argparse, json, os, glob, re

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _step_number(path):
    m = re.search(r"val_step(\d+)\.json$", path)
    return int(m.group(1)) if m else -1


def last_val_stats(results_dir, model, scene):
    pattern = os.path.join(results_dir, model, scene, "stats", "val_step*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=_step_number)
    with open(latest) as f:
        return json.load(f)


def fmt_psnr(v):
    return f"{v:.1f}"


def fmt_ssim(v):
    return f"{v:.3f}"


def fmt_lpips(v):
    return f"{v:.3f}"


COMMAND_SCENES = {"midline", "average", "total_average"}


def _avg_row(label, models, agg):
    row = label
    for model in models:
        ps, ss, ls = agg[model]["psnr"], agg[model]["ssim"], agg[model]["lpips"]
        if ps:
            row += (
                f" & {fmt_psnr(sum(ps)/len(ps))}"
                f" & {fmt_ssim(sum(ss)/len(ss))}"
                f" & {fmt_lpips(sum(ls)/len(ls))}"
            )
        else:
            row += " & -- & -- & --"
    return row + r" \\"


def build_table(models, scenes, results_dir):
    data = {}
    for model in models:
        data[model] = {}
        for scene in scenes:
            if scene not in COMMAND_SCENES:
                data[model][scene] = last_val_stats(results_dir, model, scene)

    n_metrics = 3
    metrics = ["PSNR↑", "SSIM↑", "LPIPS↓"]

    col_spec = "l" + ("".join(["c"] * n_metrics)) * len(models)
    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    model_header = r"\multirow{2.5}{*}{Scene}"
    for model in models:
        model_header += rf" & \multicolumn{{{n_metrics}}}{{c}}{{{model.replace('_', chr(92) + '_')}}}"
    lines.append(model_header + r" \\")

    cmidrules = ""
    for i in range(len(models)):
        col_start = 2 + i * n_metrics
        col_end = col_start + n_metrics - 1
        cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}}"
    lines.append(cmidrules)

    metric_header = ""
    for model in models:
        for m in metrics:
            metric_header += f" & {m}"
    lines.append(metric_header + r" \\")
    lines.append(r"\midrule")

    def empty_agg():
        return {m: {"psnr": [], "ssim": [], "lpips": []} for m in models}

    agg = empty_agg()  # resets on each `average`
    total_agg = empty_agg()  # never resets

    for scene in scenes:
        if scene == "midline":
            lines.append(r"\midrule")
        elif scene == "average":
            lines.append(_avg_row("Average", models, agg))
            agg = empty_agg()
        elif scene == "total_average":
            lines.append(_avg_row("Average", models, total_agg))
        else:
            row = scene.capitalize().replace("_", r"\_")
            for model in models:
                s = data[model][scene]
                if s is None:
                    row += " & -- & -- & --"
                else:
                    row += f' & {fmt_psnr(s["psnr"])} & {fmt_ssim(s["ssim"])} & {fmt_lpips(s["lpips"])}'
                    for acc in (agg, total_agg):
                        acc[model]["psnr"].append(s["psnr"])
                        acc[model]["ssim"].append(s["ssim"])
                        acc[model]["lpips"].append(s["lpips"])
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
        "average",
        "midline",
        "bicycle",
        "flowers",
        "garden",
        "stump",
        "treehill",
        "midline",
        "average",
        "midline",
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
        "total_average",
    ],
}

DATASETS["mn360"] = DATASETS["mip_nerf_360"]
DATASETS["ns"] = DATASETS["nerf_synthetic"]


def main():
    parser = argparse.ArgumentParser(
        description="Compile a LaTeX table from eval results"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model directory names under results/",
    )
    parser.add_argument("--scenes", nargs="+", help="Scene names")
    parser.add_argument("--datasets", nargs="+", help="Dataset names")
    parser.add_argument(
        "--results_dir",
        default=RESULTS_DIR,
        help="Path to results directory (default: ../results relative to this script)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .tex file path (default: print to stdout)",
    )
    args = parser.parse_args()

    models = args.models

    scenes = args.scenes
    if scenes is None or len(scenes) == 0:
        scenes = []
        for dataset in args.datasets:
            scenes += DATASETS[dataset]
    table = build_table(models, scenes, args.results_dir)

    if args.output:
        with open(args.output, "w") as f:
            f.write(table + "\n")
        print(f"Written to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
