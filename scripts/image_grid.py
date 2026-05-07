#!/usr/bin/env python3
"""
Produce a grid of rendered images across scenes, validation frames, and models.

Rows:    (scene, val_num) pairs  — one row per combination
Columns: model names             — use "gt" for ground truth

Ground truth is extracted from the left half of val_{i:04d}.png inside any
model's render zip (all models share the same GT pixels). You can also use
--gt_source dataset to pull from the original dataset directory instead.

Zoom and insets
---------------
  --zoom  SCENE,VAL,MODEL,X1,Y1,X2,Y2
      Crop the displayed image to the region [X1:X2, Y1:Y2] (pixel coords in
      the source image).  Use * as a wildcard for SCENE, VAL, or MODEL.

  --inset SCENE,VAL,MODEL,X1,Y1,X2,Y2,SIDE[,SCALE]
      Add a detail panel cropped from [X1:X2, Y1:Y2] placed to the left or
      right of the main image.  SCALE sets the panel height relative to the
      main image height (default 1.0).  Coordinates are always in the original
      (unzoomed) source image.  Use * as a wildcard.  Repeat the flag to add
      multiple insets — including one on each side.

  --inset_rect
      Draw a coloured rectangle on the main image at each inset's source
      region (helps readers find where the crop came from).

Examples
--------
  python scripts/image_grid.py \\
      --models gt tgs 2dgs \\
      --scenes bicycle garden \\
      --val_nums 0 \\
      --output grid.png

  python scripts/image_grid.py \\
      --models gt tgs 2dgs \\
      --scenes bonsai \\
      --val_nums 0 \\
      --zoom "bonsai,0,*,200,100,600,500" \\
      --inset "bonsai,0,*,50,50,250,200,left,0.8" \\
      --inset "bonsai,0,*,400,300,700,500,right,0.8" \\
      --inset_rect \\
      --output grid.png
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"

# Pixel gap between main image and each inset panel (in source-image pixel units).
# This gap is reserved in the column-width budget so layout is correct.
_SEP_PX = 8
_BORDER_COLOR_DEFAULT = "lightgray"
_RECT_COLOR_DEFAULT = "red"
# Line width (pts) for inset spine borders and indicate_inset rectangles
_BORDER_LW = 0.8
_RECT_LW = 1.0


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def apply_white_bg(img: np.ndarray) -> np.ndarray:
    """Composite an RGBA image onto a white background; pass RGB through."""
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        out = rgb * alpha + 255.0 * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)
    return img[:, :, :3].copy() if img.ndim == 3 else img


def _safe_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]


def _scale_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_h, 3), 255, dtype=np.uint8)
    target_w = max(1, round(target_h * w / h))
    return np.array(Image.fromarray(img).resize((target_w, target_h), Image.LANCZOS))


# ---------------------------------------------------------------------------
# Cell drawing — matplotlib inset axes
# ---------------------------------------------------------------------------

def _panel_dims(spec: dict, main_h: int) -> tuple[int, int]:
    """Return (inner_w, inner_h) in pixels for an inset panel."""
    rx1, ry1, rx2, ry2 = spec["region"]
    inner_h = max(1, round(main_h * spec.get("scale", 1.0)))
    crop_h = max(1, ry2 - ry1)
    crop_w = max(1, rx2 - rx1)
    inner_w = max(1, round(inner_h * crop_w / crop_h))
    return inner_w, inner_h


def _left_budget(insets: list[dict], ref_h: int) -> int:
    """Pixel width consumed by left insets + their trailing gaps."""
    return sum(_panel_dims(s, ref_h)[0] + _SEP_PX for s in insets)


def _right_budget(insets: list[dict], ref_h: int) -> int:
    """Pixel width consumed by leading gaps + right insets."""
    return sum(_SEP_PX + _panel_dims(s, ref_h)[0] for s in insets)


def _style_inset_ax(
    ax: plt.Axes, border_color: str, border_lw: float
) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(border_lw)


def draw_cell(
    container: plt.Axes,
    base_img: np.ndarray,
    zoom: tuple[int, int, int, int] | None,
    left_insets: list[dict],
    right_insets: list[dict],
    draw_rects: bool,
    border_color: str,
    rect_color: str,
    col_left_budget: int,
    col_main_w: int,
    col_right_budget: int,
    row_main_h: int,
) -> None:
    """
    Populate a grid cell using matplotlib inset axes.

    Column-level layout (pixels):
        [col_left_budget] [col_main_w] [col_right_budget]

    The main image is centred inside col_main_w × row_main_h.
    Insets are scaled relative to row_main_h so every row is consistent.
    All region coordinates are in the original (unzoomed) source image.
    """
    col_total_w = col_left_budget + col_main_w + col_right_budget

    main = _safe_crop(base_img, *zoom) if zoom else base_img[:, :, :3]
    actual_h, actual_w = main.shape[:2]

    # Container: invisible coordinate frame
    container.set_facecolor("white")
    container.set_xticks([])
    container.set_yticks([])
    for sp in container.spines.values():
        sp.set_visible(False)

    # Main image: centred within the column's main zone
    main_x0_px = col_left_budget + (col_main_w - actual_w) / 2
    main_y0_px = (row_main_h - actual_h) / 2
    main_ax = container.inset_axes([
        main_x0_px / col_total_w,
        main_y0_px / row_main_h,
        actual_w / col_total_w,
        actual_h / row_main_h,
    ])
    main_ax.imshow(main[:, :, :3])
    main_ax.set_xticks([])
    main_ax.set_yticks([])
    for sp in main_ax.spines.values():
        sp.set_visible(False)

    def _add_panel(spec: dict, x0_px: int) -> plt.Axes:
        iw, ih = _panel_dims(spec, row_main_h)
        y0_px = (row_main_h - ih) / 2
        ins_ax = container.inset_axes([
            x0_px / col_total_w,
            y0_px / row_main_h,
            iw / col_total_w,
            ih / row_main_h,
        ])
        crop = _safe_crop(base_img, *spec["region"])
        ins_ax.imshow(_scale_to_height(crop[:, :, :3], ih))
        _style_inset_ax(ins_ax, border_color, _BORDER_LW)
        return ins_ax

    def _indicate(spec: dict, ins_ax: plt.Axes) -> None:
        if not draw_rects:
            return
        rx1, ry1, rx2, ry2 = spec["region"]
        if zoom:
            rx1 -= zoom[0]; ry1 -= zoom[1]
            rx2 -= zoom[0]; ry2 -= zoom[1]
        main_ax.indicate_inset(
            [rx1, ry1, rx2 - rx1, ry2 - ry1],
            ins_ax,
            edgecolor=rect_color,
            facecolor="none",
            alpha=1.0,
            linewidth=_RECT_LW,
        )

    # Left panels start at x=0 and fill toward the main zone
    x_cursor = 0
    for spec in left_insets:
        ins_ax = _add_panel(spec, x_cursor)
        _indicate(spec, ins_ax)
        x_cursor += _panel_dims(spec, row_main_h)[0] + _SEP_PX

    # Right panels start at the column-level main zone boundary (not image boundary)
    x_cursor = col_left_budget + col_main_w + _SEP_PX
    for spec in right_insets:
        ins_ax = _add_panel(spec, x_cursor)
        _indicate(spec, ins_ax)
        x_cursor += _panel_dims(spec, row_main_h)[0] + _SEP_PX


# ---------------------------------------------------------------------------
# Spec parsing & matching
# ---------------------------------------------------------------------------

def _parse_inset(s: str) -> dict:
    """scene,val,model,x1,y1,x2,y2,left|right[,scale]  (* = wildcard)"""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) < 8:
        raise argparse.ArgumentTypeError(
            f"--inset expects scene,val,model,x1,y1,x2,y2,side[,scale]; got {s!r}"
        )
    side = parts[7].lower()
    if side not in ("left", "right"):
        raise argparse.ArgumentTypeError(f"inset side must be 'left' or 'right'; got {side!r}")
    return dict(
        scene=parts[0],
        val=parts[1],
        model=parts[2],
        region=(int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])),
        side=side,
        scale=float(parts[8]) if len(parts) > 8 else 1.0,
    )


def _parse_zoom(s: str) -> dict:
    """scene,val,model,x1,y1,x2,y2  (* = wildcard)"""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) < 7:
        raise argparse.ArgumentTypeError(
            f"--zoom expects scene,val,model,x1,y1,x2,y2; got {s!r}"
        )
    return dict(
        scene=parts[0],
        val=parts[1],
        model=parts[2],
        region=(int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])),
    )


def _matches(spec_val: str, actual) -> bool:
    return spec_val == "*" or spec_val == str(actual)


def _zoom_for(scene, val_num, model, zoom_specs) -> tuple | None:
    for z in reversed(zoom_specs):  # last match wins
        if _matches(z["scene"], scene) and _matches(z["val"], val_num) and _matches(z["model"], model):
            return z["region"]
    return None


def _insets_for(scene, val_num, model, inset_specs, side) -> list[dict]:
    return [
        s for s in inset_specs
        if s["side"] == side
        and _matches(s["scene"], scene)
        and _matches(s["val"], val_num)
        and _matches(s["model"], model)
    ]


# ---------------------------------------------------------------------------
# Zip / dataset image retrieval
# ---------------------------------------------------------------------------

def _find_step_zip(renders_dir: Path, step: int | None) -> Path | None:
    zips = list(renders_dir.glob("step_*.zip"))
    if not zips:
        return None
    if step is not None:
        target = renders_dir / f"step_{step}.zip"
        return target if target.exists() else None
    def _step_num(p):
        m = re.search(r"step_(\d+)\.zip$", p.name)
        return int(m.group(1)) if m else -1
    return max(zips, key=_step_num)


def _read_zip_image(zip_path: Path, filename: str) -> np.ndarray | None:
    try:
        with zipfile.ZipFile(zip_path) as z:
            if filename not in z.namelist():
                return None
            data = z.read(filename)
        return np.array(Image.open(io.BytesIO(data)))
    except Exception:
        return None


def _split_canvas(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = arr.shape[1] // 2
    return arr[:, :mid], arr[:, mid:]


def _gt_from_dataset(scene: str, val_num: int, data_dir: Path) -> np.ndarray | None:
    ns_path = data_dir / "nerf_synthetic" / scene / "val" / f"r_{val_num}.png"
    if ns_path.exists():
        return np.array(Image.open(ns_path))  # may be RGBA

    for scale_dir in ("images_4", "images_2", "images"):
        img_dir = data_dir / "mip_nerf_360" / scene / scale_dir
        if not img_dir.exists():
            continue
        imgs = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and ":Zone.Identifier" not in p.name
        )
        val_imgs = [p for i, p in enumerate(imgs) if i % 8 == 0]
        if val_num < len(val_imgs):
            return np.array(Image.open(val_imgs[val_num]))
        break

    return None


def get_image(
    model: str,
    scene: str,
    val_num: int,
    step: int | None,
    results_dir: Path,
    gt_source: str,
    data_dir: Path,
    gt_fallback_model: str | None,
) -> np.ndarray | None:
    filename = f"val_{val_num:04d}.png"

    if model == "gt":
        if gt_source == "dataset":
            img = _gt_from_dataset(scene, val_num, data_dir)
            if img is not None:
                return img

        candidates = ([gt_fallback_model] if gt_fallback_model else []) + [
            d.name
            for d in results_dir.iterdir()
            if d.is_dir() and d.name != "example_videos"
        ]
        for cand in candidates:
            renders_dir = results_dir / cand / scene / "renders"
            zip_path = _find_step_zip(renders_dir, step)
            if zip_path is None:
                continue
            arr = _read_zip_image(zip_path, filename)
            if arr is None:
                continue
            gt, _ = _split_canvas(arr)
            return gt
        return None

    renders_dir = results_dir / model / scene / "renders"
    zip_path = _find_step_zip(renders_dir, step)
    if zip_path is None:
        return None
    arr = _read_zip_image(zip_path, filename)
    if arr is None:
        return None
    _, render = _split_canvas(arr)
    return render


# ---------------------------------------------------------------------------
# Grid layout
# ---------------------------------------------------------------------------

def make_grid(
    models: list[str],
    scenes: list[str],
    val_nums: list[int],
    step: int | None,
    results_dir: Path,
    gt_source: str,
    data_dir: Path,
    zoom_specs: list[dict],
    inset_specs: list[dict],
    draw_rects: bool,
    border_color: str,
    rect_color: str,
    col_labels: dict[str, str] | None,
    row_labels: dict[tuple[str, int], str] | None,
    cell_width_in: float = 3.0,
    title: str | None = None,
    dpi: int = 150,
    output: str | None = None,
) -> None:
    rows = [(s, v) for s in scenes for v in val_nums]
    n_rows, n_cols = len(rows), len(models)
    gt_fallback = next((m for m in models if m != "gt"), None)

    # --- load base images and resolve zoom/insets per cell ---
    # cell_data[r][c] = (base_img | None, zoom, left_insets, right_insets)
    cell_data: list[list] = []
    for r, (scene, val_num) in enumerate(rows):
        row_data = []
        for c, model in enumerate(models):
            base = get_image(
                model, scene, val_num, step,
                results_dir, gt_source, data_dir, gt_fallback,
            )
            if base is not None:
                base = apply_white_bg(base)
            zoom = _zoom_for(scene, val_num, model, zoom_specs)
            left = _insets_for(scene, val_num, model, inset_specs, "left")
            right = _insets_for(scene, val_num, model, inset_specs, "right")
            row_data.append((base, zoom, left, right))
        cell_data.append(row_data)

    # --- per-cell main image dimensions ---
    def _main_wh(r: int, c: int) -> tuple[int, int]:
        base, zoom, _, _ = cell_data[r][c]
        if base is None:
            return (1, 1)
        if zoom:
            return zoom[2] - zoom[0], zoom[3] - zoom[1]
        return base.shape[1], base.shape[0]

    # Row heights and column widths driven only by the main image, not insets
    main_row_heights = [max(_main_wh(r, c)[1] for c in range(n_cols)) for r in range(n_rows)]
    main_col_widths  = [max(_main_wh(r, c)[0] for r in range(n_rows)) for c in range(n_cols)]

    # Per-column inset budgets (max across rows, using each row's canonical height)
    max_left_budgets  = [
        max(_left_budget(cell_data[r][c][2], main_row_heights[r]) for r in range(n_rows))
        for c in range(n_cols)
    ]
    max_right_budgets = [
        max(_right_budget(cell_data[r][c][3], main_row_heights[r]) for r in range(n_rows))
        for c in range(n_cols)
    ]

    col_widths  = [max_left_budgets[c] + main_col_widths[c] + max_right_budgets[c] for c in range(n_cols)]
    row_heights = main_row_heights

    # Single pixel-per-inch keeps both axes identical → preserves image aspect ratio
    px_per_in = min(col_widths) / cell_width_in
    fig_w = sum(col_widths) / px_per_in
    fig_h = sum(row_heights) / px_per_in

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": col_widths, "height_ratios": row_heights},
        squeeze=False,
    )

    if title:
        fig.suptitle(title, fontsize=14)

    for r, (scene, val_num) in enumerate(rows):
        for c, model in enumerate(models):
            ax = axes[r][c]
            base, zoom, left, right = cell_data[r][c]

            if base is None:
                ax.set_facecolor("#222222")
                ax.text(
                    0.5, 0.5, "missing",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    color="white", fontsize=9,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            else:
                draw_cell(
                    ax, base, zoom, left, right,
                    draw_rects, border_color, rect_color,
                    col_left_budget=max_left_budgets[c],
                    col_main_w=main_col_widths[c],
                    col_right_budget=max_right_budgets[c],
                    row_main_h=main_row_heights[r],
                )

            if r == 0:
                label = (col_labels or {}).get(model, model)
                ax.set_title(label, fontsize=11, pad=4)

            if c == 0:
                if row_labels and (scene, val_num) in row_labels:
                    ylabel = row_labels[(scene, val_num)]
                elif len(val_nums) > 1:
                    ylabel = f"{scene}\nval {val_num:04d}"
                else:
                    ylabel = scene
                ax.set_ylabel(
                    ylabel, fontsize=10, labelpad=6, rotation=0, ha="right", va="center"
                )

    fig.tight_layout(pad=0.3)

    if output:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an image grid across models, scenes, and val frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help='Model directory names under results/. Use "gt" for ground truth.',
    )
    parser.add_argument(
        "--scenes", nargs="+", required=True,
        help="Scene names.",
    )
    parser.add_argument(
        "--val_nums", nargs="+", type=int, default=[0], metavar="N",
        help="Validation frame indices (default: 0).",
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Training step to use (default: latest step zip).",
    )
    parser.add_argument(
        "--gt_source", choices=["zip", "dataset"], default="zip",
        help=(
            '"zip": extract GT from the left half of a model render zip (default). '
            '"dataset": load from data/mip_nerf_360 or data/nerf_synthetic.'
        ),
    )
    parser.add_argument(
        "--results_dir", default=str(RESULTS_DIR),
        help="Path to results directory.",
    )
    parser.add_argument(
        "--data_dir", default=str(DATA_DIR),
        help="Path to dataset root (used when --gt_source=dataset).",
    )
    # --- zoom / inset ---
    parser.add_argument(
        "--zoom", action="append", default=[], metavar="SPEC",
        help=(
            "Crop the main display image: SCENE,VAL,MODEL,X1,Y1,X2,Y2. "
            "Use * as wildcard. Repeat for multiple cells. "
            "Last matching spec wins."
        ),
    )
    parser.add_argument(
        "--inset", action="append", default=[], metavar="SPEC",
        help=(
            "Add a detail panel: SCENE,VAL,MODEL,X1,Y1,X2,Y2,left|right[,SCALE]. "
            "SCALE is panel height relative to main (default 1.0). "
            "Use * as wildcard. Repeat to add multiple insets per cell."
        ),
    )
    parser.add_argument(
        "--inset_rect", action="store_true",
        help="Draw a rectangle on the main image at each inset's source region.",
    )
    parser.add_argument(
        "--inset_rect_color", default=_RECT_COLOR_DEFAULT, metavar="COLOR",
        help=f"Colour of the inset source rectangle (default: {_RECT_COLOR_DEFAULT!r}).",
    )
    parser.add_argument(
        "--inset_border_color", default=_BORDER_COLOR_DEFAULT, metavar="COLOR",
        help=f"Colour of the border drawn around each inset panel (default: {_BORDER_COLOR_DEFAULT!r}).",
    )
    # --- labels / style ---
    parser.add_argument(
        "--col_labels", nargs="+", metavar="MODEL=LABEL",
        help='Override column labels, e.g. --col_labels tgs="TGS (ours)" 2dgs=2DGS',
    )
    parser.add_argument(
        "--title", default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--cell_width", type=float, default=3.0, metavar="W",
        help=(
            "Width of the narrowest column in inches (default: 3.0). "
            "Row height is derived automatically to preserve pixel aspect ratios."
        ),
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output DPI (default: 150).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output image path (PNG/PDF). Omit to display interactively.",
    )
    args = parser.parse_args()

    col_labels: dict[str, str] | None = None
    if args.col_labels:
        col_labels = {}
        for item in args.col_labels:
            k, _, v = item.partition("=")
            col_labels[k] = v or k

    zoom_specs = [_parse_zoom(s) for s in args.zoom]
    inset_specs = [_parse_inset(s) for s in args.inset]

    make_grid(
        models=args.models,
        scenes=args.scenes,
        val_nums=args.val_nums,
        step=args.step,
        results_dir=Path(args.results_dir),
        gt_source=args.gt_source,
        data_dir=Path(args.data_dir),
        zoom_specs=zoom_specs,
        inset_specs=inset_specs,
        draw_rects=args.inset_rect,
        border_color=args.inset_border_color,
        rect_color=args.inset_rect_color,
        col_labels=col_labels,
        row_labels=None,
        cell_width_in=args.cell_width,
        title=args.title,
        dpi=args.dpi,
        output=args.output,
    )


if __name__ == "__main__":
    main()
