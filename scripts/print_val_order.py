#!/usr/bin/env python3
"""Print the val sort order from the dataset cache.

Shows the mapping between rendered val image index (val_XXXX.png in render zips)
and the original dataset val image index.

Usage:
    python scripts/print_val_order.py --data_dir data/nerf_synthetic/chair
    python scripts/print_val_order.py --data_dir data/mip_nerf_360/garden
    python scripts/print_val_order.py --data_dir data/nerf_synthetic/chair --inverse
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

_VAL_ORDER_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "val_order"


def _cache_path(data_dir: Path, split_id: str) -> Path:
    params = {"data_dir": str(data_dir.resolve()), "split_id": split_id}
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    return _VAL_ORDER_CACHE_DIR / f"{data_dir.name}_{key}.npy"


def _detect_split_id(data_dir: Path) -> str:
    if (data_dir / "transforms_val.json").exists() or (data_dir / "val").exists():
        return "blender_val"
    return "8"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the val sort order from the dataset cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Scene data directory (e.g. data/nerf_synthetic/chair or data/mip_nerf_360/garden).",
    )
    parser.add_argument(
        "--split_id",
        default=None,
        help='Val split identifier. Auto-detected if omitted ("blender_val" or "8").',
    )
    parser.add_argument(
        "--inverse",
        action="store_true",
        help="Also print the inverse mapping: original val index → render index.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split_id = args.split_id or _detect_split_id(data_dir)
    cache_path = _cache_path(data_dir, split_id)

    if not cache_path.exists():
        print(f"No cache found: {cache_path}")
        print(f"  data_dir  : {data_dir.resolve()}")
        print(f"  split_id  : {split_id!r}")
        return 1

    order = np.load(cache_path)
    n = len(order)

    print(f"Val sort order for '{data_dir.name}'  (split_id={split_id!r})")
    print(f"  Cache : {cache_path}")
    print(f"  N     : {n} val images")
    print()
    print(f"  {'Render #':>9}  →  Original val #")
    print(f"  {'-'*9}     {'-'*14}")
    for render_idx, orig_idx in enumerate(order):
        print(f"  {render_idx:>9}  →  {int(orig_idx)}")

    if args.inverse:
        inv_order = np.empty(n, dtype=order.dtype)
        inv_order[order] = np.arange(n, dtype=order.dtype)
        print()
        print(f"  {'Original val #':>14}  →  Render #")
        print(f"  {'-'*14}     {'-'*8}")
        for orig_idx in range(n):
            print(f"  {orig_idx:>14}  →  {int(inv_order[orig_idx])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())