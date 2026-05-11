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
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"

# Pixel gap between main image and each inset panel (in source-image pixel units).
# This gap is reserved in the column-width budget so layout is correct.
_SEP_PX = 8
_BORDER_COLOR_DEFAULT = "red"
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


def _style_inset_ax(ax: plt.Axes, border_color: str, border_lw: float) -> None:
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
    col_width: int,
    row_main_h: int,
) -> None:
    """
    Populate a grid cell using matplotlib inset axes.

    The full group (left insets + main image + right insets) is scaled uniformly
    so that it fills col_width exactly, then centred vertically in row_main_h.
    All region coordinates are in the original (unzoomed) source image.
    """
    main = _safe_crop(base_img, *zoom) if zoom else base_img[:, :, :3]
    actual_h, actual_w = main.shape[:2]

    # Native group dimensions at this cell's own pixel scale
    left_dims = [_panel_dims(s, actual_h) for s in left_insets]
    right_dims = [_panel_dims(s, actual_h) for s in right_insets]
    native_group_w = (
        sum(iw + _SEP_PX for iw, _ in left_dims)
        + actual_w
        + sum(_SEP_PX + iw for iw, _ in right_dims)
    )
    native_group_h = max([actual_h] + [ih for _, ih in left_dims + right_dims])

    # Scale group to fill the column width exactly
    cscale = col_width / max(1, native_group_w)
    disp_group_h = native_group_h * cscale
    # Vertical offset to centre group in the row
    y_top = (row_main_h - disp_group_h) / 2

    # Container: invisible coordinate frame
    container.set_facecolor("white")
    container.set_xticks([])
    container.set_yticks([])
    for sp in container.spines.values():
        sp.set_visible(False)

    def _place(
        img: np.ndarray, x0: float, native_w: float, native_h: float
    ) -> plt.Axes:
        disp_w = native_w * cscale
        disp_h = native_h * cscale
        y0 = y_top + (disp_group_h - disp_h) / 2
        ax = container.inset_axes(
            [
                x0 / col_width,
                y0 / row_main_h,
                disp_w / col_width,
                disp_h / row_main_h,
            ]
        )
        resized = _scale_to_height(img[:, :, :3], max(1, round(disp_h)))
        ax.imshow(resized)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def _indicate(spec: dict, ins_ax: plt.Axes) -> None:
        if not draw_rects:
            return
        rx1, ry1, rx2, ry2 = spec["region"]
        if zoom:
            rx1 -= zoom[0]
            ry1 -= zoom[1]
            rx2 -= zoom[0]
            ry2 -= zoom[1]
        # imshow data coords are in resampled-image pixels, so scale by cscale
        _, connectors = main_ax.indicate_inset(
            [rx1 * cscale, ry1 * cscale, (rx2 - rx1) * cscale, (ry2 - ry1) * cscale],
            ins_ax,
            edgecolor=rect_color,
            facecolor="none",
            alpha=1.0,
            linewidth=_RECT_LW,
        )
        # Clip connectors to the main image so they never draw over the inset
        for conn in connectors:
            if conn is not None:
                conn.set_clip_on(True)
                conn.set_clip_path(main_ax.patch)

    # --- left insets ---
    x_cursor = 0.0
    pending_indicate: list[tuple[dict, plt.Axes]] = []
    for spec, (iw, ih) in zip(left_insets, left_dims):
        crop = _safe_crop(base_img, *spec["region"])
        ins_ax = _place(crop, x_cursor, iw, ih)
        _style_inset_ax(ins_ax, border_color, _BORDER_LW)
        pending_indicate.append((spec, ins_ax))
        x_cursor += iw * cscale + _SEP_PX * cscale

    # --- main image (main_ax must exist before _indicate is called) ---
    main_ax = _place(main, x_cursor, actual_w, actual_h)
    for sp in main_ax.spines.values():
        sp.set_visible(False)
    x_cursor += actual_w * cscale

    # indicate left insets now that main_ax exists
    for spec, ins_ax in pending_indicate:
        _indicate(spec, ins_ax)

    # --- right insets ---
    for spec, (iw, ih) in zip(right_insets, right_dims):
        x_cursor += _SEP_PX * cscale
        crop = _safe_crop(base_img, *spec["region"])
        ins_ax = _place(crop, x_cursor, iw, ih)
        _style_inset_ax(ins_ax, border_color, _BORDER_LW)
        _indicate(spec, ins_ax)
        x_cursor += iw * cscale


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
        raise argparse.ArgumentTypeError(
            f"inset side must be 'left' or 'right'; got {side!r}"
        )
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
        if (
            _matches(z["scene"], scene)
            and _matches(z["val"], val_num)
            and _matches(z["model"], model)
        ):
            return z["region"]
    return None


def _insets_for(scene, val_num, model, inset_specs, side) -> list[dict]:
    return [
        s
        for s in inset_specs
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
            p
            for p in img_dir.iterdir()
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
    row_model_overrides: dict[tuple[str, int], list[str]] | None = None,
    cell_width_in: float = 3.0,
    fig_width_in: float | None = None,
    cell_gap: float = 0.3,
    font_size: float = 11.0,
    title: str | None = None,
    dpi: int = 150,
    output: str | None = None,
) -> None:
    rows = list(zip(scenes, val_nums))

    # Columns are always the canonical --models list; row overrides substitute
    # model names at the same column positions (validated to have equal length).
    all_models: list[str] = models
    n_rows, n_cols = len(rows), len(all_models)

    # --- load base images and resolve zoom/insets per cell ---
    # cell_data[r][c] = (base_img | None, zoom, left_insets, right_insets)
    cell_data: list[list] = []
    for r, (scene, val_num) in enumerate(rows):
        fetch_models = (row_model_overrides or {}).get((scene, val_num), models)
        gt_fallback = next((m for m in fetch_models if m != "gt"), None)
        row_data = []
        for c in range(n_cols):
            fetch_model = fetch_models[c]
            base = get_image(
                fetch_model,
                scene,
                val_num,
                step,
                results_dir,
                gt_source,
                data_dir,
                gt_fallback,
            )
            if base is not None:
                base = apply_white_bg(base)
            zoom = _zoom_for(scene, val_num, fetch_model, zoom_specs)
            left = _insets_for(scene, val_num, fetch_model, inset_specs, "left")
            right = _insets_for(scene, val_num, fetch_model, inset_specs, "right")
            row_data.append((base, zoom, left, right))
        cell_data.append(row_data)

    # --- per-cell main image dimensions ---
    def _main_wh(r: int, c: int) -> tuple[int, int]:
        base, zoom, _, _ = cell_data[r][c]
        if base is None:
            return (1, 1)
        if zoom:
            img_h, img_w = base.shape[:2]
            x1, y1, x2, y2 = zoom
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            return max(1, x2 - x1), max(1, y2 - y1)
        return base.shape[1], base.shape[0]

    # Native group dimensions per cell: the full (insets + main image) group
    # at the cell's own pixel scale, before any column-level scaling.
    def _cell_group(r: int, c: int) -> tuple[int, int]:
        """Return (native_group_w, native_group_h) in source pixels."""
        mw, mh = _main_wh(r, c)
        left_insets = cell_data[r][c][2]
        right_insets = cell_data[r][c][3]
        gw = _left_budget(left_insets, mh) + mw + _right_budget(right_insets, mh)
        all_insets = left_insets + right_insets
        gh = max([mh] + [_panel_dims(s, mh)[1] for s in all_insets])
        return gw, gh

    cell_gw = [[_cell_group(r, c)[0] for c in range(n_cols)] for r in range(n_rows)]
    cell_gh = [[_cell_group(r, c)[1] for c in range(n_cols)] for r in range(n_rows)]

    # Column width = max native group width across rows (ignore degenerate cells).
    col_widths = [max(cell_gw[r][c] for r in range(n_rows)) for c in range(n_cols)]

    # Each cell's scale factor stretches its group to fill the column width.
    def _cscale(r: int, c: int) -> float:
        return col_widths[c] / max(1, cell_gw[r][c])

    # Row height = max scaled group height across columns.
    row_heights = [
        max(cell_gh[r][c] * _cscale(r, c) for c in range(n_cols)) for r in range(n_rows)
    ]

    # Single pixel-per-inch keeps both axes identical → preserves image aspect ratio.
    # fig_width_in fixes the total figure width; cell_width_in is the fallback that
    # sizes the narrowest column to a given width.
    if fig_width_in is not None:
        px_per_in = sum(col_widths) / fig_width_in
    else:
        # Exclude degenerate (all-missing) columns from the scale reference to avoid
        # astronomically large figures when a model directory doesn't exist.
        ref_widths = [w for w in col_widths if w > 1] or col_widths
        px_per_in = min(ref_widths) / cell_width_in
    fig_w = sum(col_widths) / px_per_in
    fig_h = sum(row_heights) / px_per_in

    fig, axes_arr = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": col_widths, "height_ratios": row_heights},
        squeeze=False,
    )
    axes: list[list[plt.Axes]] = axes_arr.tolist()

    if title:
        fig.suptitle(title, fontsize=font_size + 3)

    for r, (scene, val_num) in enumerate(rows):
        for c, model in enumerate(all_models):
            ax = axes[r][c]
            base, zoom, left, right = cell_data[r][c]

            if base is None:
                ax.set_facecolor("white")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            else:
                draw_cell(
                    ax,
                    base,
                    zoom,
                    left,
                    right,
                    draw_rects,
                    border_color,
                    rect_color,
                    col_width=col_widths[c],
                    row_main_h=row_heights[r],
                )

            if r == 0:
                label = (col_labels or {}).get(model, model)
                ax.set_title(label, fontsize=font_size, pad=4)

            if c == 0:
                if row_labels and (scene, val_num) in row_labels:
                    ylabel = row_labels[(scene, val_num)]
                elif len(set(scenes)) < len(scenes):
                    ylabel = f"{scene}\nval {val_num:04d}"
                else:
                    ylabel = scene
                ax.set_ylabel(
                    ylabel,
                    fontsize=font_size,
                    labelpad=6,
                    rotation=90,
                    ha="center",
                    va="center",
                )

    fig.tight_layout(pad=cell_gap, h_pad=0, w_pad=0)

    # Zero out wspace so adjacent columns share a border — no whitespace between
    # inset panels.  hspace is kept from tight_layout because setting it to zero
    # displaces inset_axes / indicate_inset artists and causes bbox_inches='tight'
    # to balloon the saved image.
    sp = fig.subplotpars
    fig.subplots_adjust(
        left=sp.left,
        right=sp.right,
        top=sp.top,
        bottom=sp.bottom,
        wspace=0,
        hspace=sp.hspace,
    )

    if fig_width_in is not None:
        # tight_layout shrinks the subplot region in both x and y to make room
        # for labels.  Rescale x so the axes content area is exactly fig_width_in
        # wide, and independently rescale y so the content height equals the
        # original intended height (fig_h).
        x_frac = sp.right - sp.left
        y_frac = sp.top - sp.bottom
        if x_frac > 0 and y_frac > 0:
            fig.set_size_inches(fig_width_in / x_frac, fig_h / y_frac)

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
        "--models",
        nargs="+",
        required=True,
        help='Model directory names under results/. Use "gt" for ground truth.',
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        required=True,
        help="Scene names.",
    )
    parser.add_argument(
        "--val_nums",
        nargs="+",
        type=int,
        default=[0],
        metavar="N",
        help="Validation frame indices (default: 0).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Training step to use (default: latest step zip).",
    )
    parser.add_argument(
        "--gt_source",
        choices=["zip", "dataset"],
        default="zip",
        help=(
            '"zip": extract GT from the left half of a model render zip (default). '
            '"dataset": load from data/mip_nerf_360 or data/nerf_synthetic.'
        ),
    )
    parser.add_argument(
        "--results_dir",
        default=str(RESULTS_DIR),
        help="Path to results directory.",
    )
    parser.add_argument(
        "--data_dir",
        default=str(DATA_DIR),
        help="Path to dataset root (used when --gt_source=dataset).",
    )
    # --- zoom / inset ---
    parser.add_argument(
        "--zoom",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Crop the main display image: SCENE,VAL,MODEL,X1,Y1,X2,Y2. "
            "Use * as wildcard. Repeat for multiple cells. "
            "Last matching spec wins."
        ),
    )
    parser.add_argument(
        "--inset",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Add a detail panel: SCENE,VAL,MODEL,X1,Y1,X2,Y2,left|right[,SCALE]. "
            "SCALE is panel height relative to main (default 1.0). "
            "Use * as wildcard. Repeat to add multiple insets per cell."
        ),
    )
    parser.add_argument(
        "--inset_rect",
        action="store_true",
        help="Draw a rectangle on the main image at each inset's source region.",
    )
    parser.add_argument(
        "--inset_rect_color",
        default=_RECT_COLOR_DEFAULT,
        metavar="COLOR",
        help=f"Colour of the inset source rectangle (default: {_RECT_COLOR_DEFAULT!r}).",
    )
    parser.add_argument(
        "--inset_border_color",
        default=_BORDER_COLOR_DEFAULT,
        metavar="COLOR",
        help=f"Colour of the border drawn around each inset panel (default: {_BORDER_COLOR_DEFAULT!r}).",
    )
    # --- labels / style ---
    parser.add_argument(
        "--col_labels",
        nargs="+",
        metavar="MODEL=LABEL",
        help='Override column labels, e.g. --col_labels tgs="TGS (ours)" 2dgs=2DGS',
    )
    parser.add_argument(
        "--row_labels",
        nargs="+",
        metavar="SCENE,VAL=LABEL",
        help='Override row labels, e.g. --row_labels counter,9="Counter"',
    )
    parser.add_argument(
        "--row_models",
        action="append",
        default=[],
        metavar="SCENE,VAL,model1,model2,...",
        help=(
            "Override models for a specific row: SCENE,VAL,model1,model2,... "
            "Rows without an override use --models. Repeat for multiple rows."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--fig_width",
        type=float,
        default=160.0,
        metavar="MM",
        help=(
            "Total figure width in mm (default: 160, i.e. A4 with 25 mm margins). "
            "All columns are scaled proportionally to fit this width. "
            "When set, --cell_width is ignored."
        ),
    )
    parser.add_argument(
        "--cell_width",
        type=float,
        default=3.0,
        metavar="W",
        help=(
            "Width of the narrowest column in inches (default: 3.0). "
            "Used only when --fig_width is not set. "
            "Row height is derived automatically to preserve pixel aspect ratios."
        ),
    )
    parser.add_argument(
        "--cell_gap",
        type=float,
        default=0.3,
        metavar="G",
        help=(
            "Spacing between cells in font-size units (default: 0.3). "
            "Set to 0 for no gap between images."
        ),
    )
    parser.add_argument(
        "--font_size",
        type=float,
        default=11.0,
        metavar="PT",
        help="Font size in points for column/row labels (default: 11). Title is font_size + 3.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output image path (PNG/PDF). Omit to display interactively.",
    )
    args = parser.parse_args()

    col_labels: dict[str, str] | None = None
    if args.col_labels:
        col_labels = {}
        for item in args.col_labels:
            k, _, v = item.partition("=")
            col_labels[k] = v or k

    row_labels: dict[tuple[str, int], str] | None = None
    if args.row_labels:
        row_labels = {}
        for item in args.row_labels:
            key_part, _, label = item.partition("=")
            parts = key_part.rsplit(",", 1)
            if len(parts) != 2:
                raise SystemExit(
                    f"--row_labels: expected SCENE,VAL=LABEL, got {item!r}"
                )
            row_labels[(parts[0], int(parts[1]))] = label

    row_model_overrides: dict[tuple[str, int], list[str]] | None = None
    if args.row_models:
        row_model_overrides = {}
        for item in args.row_models:
            parts = item.split(",")
            if len(parts) < 3:
                raise SystemExit(
                    f"--row_models: expected SCENE,VAL,model1,..., got {item!r}"
                )
            override_models = [m.strip() for m in parts[2:]]
            if len(override_models) != len(args.models):
                raise SystemExit(
                    f"--row_models {parts[0]},{parts[1]}: got {len(override_models)} model(s) "
                    f"but --models has {len(args.models)}. Counts must match so columns align."
                )
            row_model_overrides[(parts[0], int(parts[1]))] = override_models

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
        row_labels=row_labels,
        row_model_overrides=row_model_overrides,
        cell_width_in=args.cell_width,
        fig_width_in=args.fig_width / 25.4,
        cell_gap=args.cell_gap,
        font_size=args.font_size,
        title=args.title,
        dpi=args.dpi,
        output=args.output,
    )


if __name__ == "__main__":
    main()
