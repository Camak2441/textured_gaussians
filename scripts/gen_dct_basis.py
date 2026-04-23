"""
Generate a grid image of all DCT-II basis functions up to a given block size.

Usage:
    python scripts/gen_dct_basis.py --size 8 --tile-res 64 --out dct_basis.png
"""

import argparse
import math
import numpy as np
from PIL import Image


def dct_basis(u: int, v: int, N: int, res: int) -> np.ndarray:
    i = np.arange(res, dtype=np.float32)
    cos_u = np.cos(math.pi * u * (i + 0.5) / res)
    cos_v = np.cos(math.pi * v * (i + 0.5) / res)
    return cos_u[:, None] * cos_v[None, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=int,
        default=8,
        help="DCT block size N (generates N×N basis images)",
    )
    parser.add_argument(
        "--tile-res", type=int, default=64, help="pixel resolution of each tile"
    )
    parser.add_argument("--out", type=str, default="dct_basis.png")
    args = parser.parse_args()

    N, res = args.size, args.tile_res
    gap = 2
    canvas_w = N * res + (N + 1) * gap
    canvas_h = N * res + (N + 1) * gap
    canvas = np.full((canvas_h, canvas_w), 0.5, dtype=np.float32)

    for u in range(N):
        for v in range(N):
            tile = dct_basis(u, v, N, res) / 2.0 + 0.5
            y0 = gap + u * (res + gap)
            x0 = gap + v * (res + gap)
            canvas[y0 : y0 + res, x0 : x0 + res] = tile

    img = Image.fromarray((canvas * 255).clip(0, 255).astype(np.uint8), mode="L")
    img.save(args.out)
    print(f"Saved {N}×{N} DCT basis grid ({canvas_w}×{canvas_h} px) → {args.out}")


if __name__ == "__main__":
    main()
