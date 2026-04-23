"""
Fix Windows-style backslash paths in Blender NeRF dataset JSON files.

Rewrites all `file_path` values in transforms_*.json files in-place,
replacing backslashes with forward slashes.

Usage:
    python scripts/fix_blender_paths.py <dataset_dir>

Example:
    python scripts/fix_blender_paths.py data/custom/my_scene
"""

import argparse
import json
from pathlib import Path


def fix_transforms(json_path: Path) -> int:
    with open(json_path) as f:
        data = json.load(f)

    changed = 0
    for frame in data.get("frames", []):
        original = frame.get("file_path", "")
        fixed = original.replace("\\", "/")
        if fixed != original:
            frame["file_path"] = fixed
            changed += 1

    if changed:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"  {json_path.name}: fixed {changed} path(s)")
    else:
        print(f"  {json_path.name}: already clean")

    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="path to Blender dataset directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"Not a directory: {dataset_dir}")

    json_files = sorted(dataset_dir.glob("transforms_*.json"))
    if not json_files:
        raise SystemExit(f"No transforms_*.json files found in {dataset_dir}")

    total = 0
    for json_path in json_files:
        total += fix_transforms(json_path)

    print(f"Done — {total} path(s) fixed across {len(json_files)} file(s).")


if __name__ == "__main__":
    main()
