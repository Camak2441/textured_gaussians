#!/usr/bin/env python3
"""
Compile a LaTeX table from evaluation results.

Usage:
  python scripts/compile_latex_table.py \
    --models 2dgs tgs \
    --scenes chair drums ficus hotdog lego materials mic ship \
    --results_dir results
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


MODEL_NAMES = {
    "2dgs": "2DGS (10000 Splats, Random)",
    "2dgs_g2000": "2DGS (2000 Splats, Random)",
    "2dgss_g1966_oquad1-1000_swc08": "2DGSS (1966 Splats, Random)",
    "2dgss_g1999_oquad1-1000_swc08": "2DGSS (1999 Splats, Random)",
    "2dgss_g9833_oquad1-1000_swc08": "2DGSS (9833 Splats, Random)",
    "2dgss_g9999_oquad1-1000_swc08": "2DGSS (9999 Splats, Random)",
    "2dgs_sfm": "2DGS (10000 Splats, SfM)",
    "2dgss_g9833_sfm_oquad1-1000_swc08": "2DGSS (9833 Splats, SfM)",
    "2dgss_g9999_sfm_oquad1-1000_swc08": "2DGSS (9999 Splats, SfM)",
    "tgs": "TGS (10000 Splats, Random)",
    "tgs_g2000": "TGS (2000 Splats, Random)",
    "tgs_b2": "TGS w/ TexGrad (10000 Splats, Random)",
    "tgs_b2_g2000": "TGS w/ TexGrad (2000 Splats, Random)",
    "tgss4_b2_g1999_ot01-0_ott03-0_sgc02_swc08_po_pswc08": "TGSS (1999 Splats, Random)",
    "tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_po_pswc08": "TGSS (9999 Splats, Random)",
    "tgs_psfm": "TGS (10000 Splats, SfM)",
    "tgs_b2_psfm": "TGS w/ TexGrad (10000 Splats, SfM)",
    "tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_psfm_po_pswc08": "TGSS (9999 Splats, SfM)",
}


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


def build_table(
    models,
    scenes,
    results_dir,
    font_size="small",
    smaller_models=True,
    smaller_metrics=True,
):
    font_size_index = get_font_size_index(font_size)
    base_font_size = LATEX_FONT_SIZES[font_size_index]
    smaller_font_index = max(0, font_size_index - 1)
    smaller_font_size = LATEX_FONT_SIZES[smaller_font_index]

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

    model_header = r"\multicolumn{1}{c}{Model}"
    for model in models:
        model_header += rf" & \multicolumn{{{n_metrics}}}{{c}}"
        model_header += "{"
        if smaller_models:
            model_header += rf"\{smaller_font_size} "
        model_header += MODEL_NAMES.get(model, model.replace("_", r"\_")) + "}"
    lines.append(model_header + r" \\")

    cmidrules = r"\cmidrule{1-1}"
    for i in range(len(models)):
        col_start = 2 + i * n_metrics
        col_end = col_start + n_metrics - 1
        cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}}"
    lines.append(cmidrules)

    metric_header = r"\multicolumn{1}{c}{Scene $|$ Metric}"
    for model in models:
        for m in metrics:
            metric_header += " & "
            if smaller_metrics:
                metric_header += rf"\{smaller_font_size} "
            metric_header += m
    lines.append(metric_header + r" \\")
    lines.append(r"\midrule")

    def empty_agg():
        return {m: {"psnr": [], "ssim": [], "lpips": []} for m in models}

    agg = empty_agg()  # resets on each `average`
    total_agg = empty_agg()  # never resets

    rename_next = None

    for scene in scenes:
        if scene == "midline":
            lines.append(r"\midrule")
        elif scene == "average":
            name = "Average"
            if rename_next:
                name = rename_next
                rename_next = None
            lines.append(_avg_row(name, models, agg))
            agg = empty_agg()
        elif scene == "total_average":
            name = "Average"
            if rename_next:
                name = rename_next
                rename_next = None
            lines.append(_avg_row(name, models, total_agg))
        elif scene.startswith("rename_to_"):
            rename_next = scene[len("rename_to_") :]
        else:
            name = scene.capitalize().replace("_", r"\_")
            if rename_next:
                name = rename_next
                rename_next = None
            row = name
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
}

DATASETS["mn360"] = DATASETS["mip_nerf_360"]
DATASETS["ns"] = DATASETS["nerf_synthetic"]


def main():
    parser = argparse.ArgumentParser(
        description="Compile a LaTeX table from eval results"
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        required=True,
        help="Model directory names under results/",
    )
    parser.add_argument("--scenes", "-s", nargs="+", help="Scene names")
    parser.add_argument("--datasets", "-ds", nargs="+", help="Dataset names")
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
    parser.add_argument("--smaller-models", "-sml", action="store_true")
    parser.add_argument("--smaller-metrics", "-smt", action="store_true")
    args = parser.parse_args()

    models = args.models

    scenes = args.scenes
    if scenes is None or len(scenes) == 0:
        scenes = []
        for dataset in args.datasets:
            scenes += DATASETS[dataset]
    table = build_table(
        models,
        scenes,
        args.results_dir,
        font_size=args.font_size,
        smaller_models=args.smaller_models,
        smaller_metrics=args.smaller_metrics,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(table + "\n")
        print(f"Written to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
