#!/usr/bin/bash

python image_grid.py \
    --dpi 900 --cell_gap 0 --font_size 9 --inset_rect \
    --border_lw 0.6 --rect_lw 0.4 \
    --models gt 2dgs_sfm 2dgs_sfm_oquad1-1000 2dgss_g9833_sfm_oquad1-1000_swc08 \
    --col_labels "gt=Ground Truth" "2dgs_sfm=2DGS" "2dgs_sfm_oquad1-1000=2DGS w/ OpacLoss" "2dgss_g9833_sfm_oquad1-1000_swc08=2DG-SS" \
    --scenes counter room chair lego mic \
    --val_nums 10 3 30 50 50 \
    --val_ordering original \
    --row_models "chair,30,gt,2dgs_g2000,2dgs_g2000_oquad1-1000,2dgss_g1966_oquad1-1000_swc08" \
    --row_models "lego,50,gt,2dgs_g2000,2dgs_g2000_oquad1-1000,2dgss_g1966_oquad1-1000_swc08" \
    --row_models "mic,50,gt,2dgs_g2000,2dgs_g2000_oquad1-1000,2dgss_g1966_oquad1-1000_swc08" \
    --output ../results/images/2dgs.png
