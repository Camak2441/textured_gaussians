#!/usr/bin/bash

python image_grid.py \
    --dpi 900 --cell_gap 0 --font_size 9 --inset_rect \
    --border_lw 0.6 --rect_lw 0.4 \
    --models gt 2dgs 2dgss_g9833_oquad1-1000_swc08 \
    --col_labels "gt=Ground Truth" "2dgs=2DGS" "2dgss_g9833_oquad1-1000_swc08=2DG-SS" \
    --scenes mic \
    --val_nums 5 \
    --val_ordering original \
    --output ../results/images/2dgs2_test.png
