#!/usr/bin/bash

python image_grid.py \
    --dpi 500 --cell_gap 0 --font_size 9 --inset_rect \
    --fig_width 257 \
    --border_lw 0.6 --rect_lw 0.4 \
    --models gt \
    2dgs_g2000 2dgs_g2000_oquad1-1000 2dgss_g1966_oquad1-1000_swc08 \
    2dgs 2dgs_oquad1-1000 2dgss_g9833_oquad1-1000_swc08 \
    --col_labels "gt=Ground Truth" \
    "2dgs_g2000=2DGS 2k" "2dgs_g2000_oquad1-1000=2DGS 2k w/ OpacLoss" "2dgss_g1966_oquad1-1000_swc08=2DG-SS 2k" \
    "2dgs=2DGS 10k" "2dgs_oquad1-1000=2DGS 10k w/ OpacLoss" "2dgss_g9833_oquad1-1000_swc08=2DG-SS 10k" \
    --scenes ship \
    --val_nums 44 \
    --val_ordering original \
    --zoom "ship,44,*,50,125,775,800" \
    --inset "ship,44,*,300,200,600,600,right,1.0" \
    --output ../results/images/2dgs_ns_test.png
