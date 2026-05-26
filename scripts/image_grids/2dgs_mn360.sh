#!/usr/bin/bash

python image_grid.py \
    --dpi 900 --cell_gap 0 --font_size 9 --inset_rect \
    --fig_width 257 \
    --border_lw 0.6 --rect_lw 0.4 \
    --models gt \
    2dgs 2dgs_oquad1-1000 2dgss_g9833_oquad1-1000_swc08 \
    2dgs_sfm 2dgs_sfm_oquad1-1000 2dgss_g9833_sfm_oquad1-1000_swc08 \
    --col_labels "gt=Ground Truth" \
    "2dgs=2DGS" "2dgs_oquad1-1000=2DGS w/ OpacLoss" "2dgss_g9833_oquad1-1000_swc08=2DG-SS" \
    "2dgs_sfm=2DGS w/ SfM" "2dgs_sfm_oquad1-1000=2DGS w/ SfM & OpacLoss" "2dgss_g9833_sfm_oquad1-1000_swc08=2DG-SS w/ SfM" \
    --scenes garden kitchen room \
    --val_nums 3 21 3 \
    --val_ordering original \
    --row_labels "garden,3=Garden" "kitchen,21=Kitchen" "room,3=Room" \
    --inset "garden,3,*,550,400,800,800,right,1.0" \
    --inset "kitchen,21,*,0,0,200,300,left,1.0" \
    --inset "kitchen,21,*,400,0,600,300,right,1.0" \
    --inset "room,3,*,600,0,780,250,right,1.0" \
    --output ../results/images/2dgs_mn360.png
