#!/usr/bin/bash

python image_grid.py \
    --dpi 900 --cell_gap 0 --font_size 9 --inset_rect \
    --border_lw 0.6 --rect_lw 0.4 \
    --models gt \
    2dss_g2000_k2 2dss_g2000_k3 2dss_g2000_k5 2dss_g2000_k10 \
    --col_labels "gt=Ground Truth" \
    "2dss_g2000_k2=2D-SS k=2" "2dss_g2000_k3=2D-SS k=3" \
    "2dss_g2000_k5=2D-SS k=5" "2dss_g2000_k10=2D-SS k=10" \
    --scenes chair chair chair \
    --val_nums 0 9 13 \
    --val_ordering original \
    --row_labels "chair,0=Chair (0)" "chair,9=Chair (9)" "chair,13=Chair (13)" \
    --output ../results/images/2dss_ns.png
