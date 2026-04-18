#!/usr/bin/bash
# Zip all existing texture step folders and render folders under results/.
set -euo pipefail

RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"

echo "=== Zipping texture step folders ==="
find "$RESULTS_DIR" -type d -regex '.*/textures[0-9]*x[0-9]*/step_[0-9]*' | while read -r step_dir; do
    zip_path="${step_dir}.zip"
    if [ -f "$zip_path" ]; then
        echo "Skipping $step_dir (zip already exists)"
        continue
    fi
    echo "Zipping $step_dir -> $zip_path"
    (cd "$step_dir" && zip -0 -r "$zip_path" .)
    rm -rf "$step_dir"
    echo "Done: $zip_path"
done

echo "=== Zipping render folders ==="
find "$RESULTS_DIR" -type d -name "renders" | while read -r render_dir; do
    # Skip if already empty or only contains zip files
    n_loose=$(find "$render_dir" -maxdepth 1 -name "*.png" | wc -l)
    if [ "$n_loose" -eq 0 ]; then
        echo "Skipping $render_dir (no loose PNGs)"
        continue
    fi

    # Group PNGs by step: files with a _<step>.png suffix, plus un-suffixed val images
    # Collect all distinct step numbers from filenames
    steps=$(find "$render_dir" -maxdepth 1 -name "val_*_*.png" \
        | grep -oP '(?<=_)\d+(?=\.png$)' | sort -u)

    if [ -z "$steps" ]; then
        # No step-tagged files — zip everything into renders.zip
        zip_path="${render_dir}.zip"
        if [ -f "$zip_path" ]; then
            echo "Skipping $render_dir (renders.zip already exists)"
        else
            echo "Zipping $render_dir -> $zip_path"
            (cd "$render_dir" && zip -0 -r "$zip_path" *.png)
            find "$render_dir" -maxdepth 1 -name "*.png" -delete
            echo "Done: $zip_path"
        fi
        continue
    fi

    for step in $steps; do
        zip_path="${render_dir}/step_${step}.zip"
        if [ -f "$zip_path" ]; then
            echo "Skipping step $step in $render_dir (zip already exists)"
            continue
        fi
        echo "Zipping step $step renders in $render_dir -> $zip_path"
        # Collect files for this step: val_NNNN.png (un-tagged) + val_NNNN_*_<step>.png
        mapfile -t files < <(find "$render_dir" -maxdepth 1 \( \
            -name "val_????_*_${step}.png" -o -name "val_????.png" \))
        if [ ${#files[@]} -eq 0 ]; then
            continue
        fi
        (cd "$render_dir" && zip -0 "$zip_path" "${files[@]##*/}")
        # Only delete the step-tagged files; val_NNNN.png may be shared across steps
        find "$render_dir" -maxdepth 1 -name "val_????_*_${step}.png" -delete
        echo "Done: $zip_path"
    done

    # Delete any remaining un-suffixed val_NNNN.png that were included in all step zips
    find "$render_dir" -maxdepth 1 -name "val_????.png" -delete
done

echo "=== Done ==="