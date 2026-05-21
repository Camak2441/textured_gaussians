#!/usr/bin/bash

python image_grid.py \
    --dpi 600 --cell_gap 0 --font_size 9 --inset_rect \
    --models gt 2dgs_sfm 2dgss_g9833_sfm_oquad1-1000_swc08 \
    --col_labels "gt=Ground Truth" "2dgs_sfm=2DGS" "2dgss_g9833_sfm_oquad1-1000_swc08=2DG-SS" \
    --scenes ship \
    --val_nums 40 \
    --val_ordering original \
    --row_models "ship,40,gt,2dgs_g2000,2dgss_g1966_oquad1-1000_swc08" \
    --zoom "lego,5,*,25,100,750,650" \
    --inset "lego,5,*,100,250,200,550,left,1.0" \
    --inset "lego,5,*,300,175,450,525,right,1.0" \
    --output ../results/images/2dgs_test.png
