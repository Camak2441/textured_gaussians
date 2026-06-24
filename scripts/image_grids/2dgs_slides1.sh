#!/usr/bin/bash

python image_grid.py \
    --dpi 900 --cell_gap 0 --font_size 9 --inset_rect \
    --fig_width 257 \
    --border_lw 0.6 --rect_lw 0.4 \
    --models gt \
    2dgs 2dgs_oquad1-1000 2dgss_g9833_oquad1-1000_swc08 \
    --col_labels "gt=Ground Truth" \
    "2dgs=2DGS" "2dgs_oquad1-1000=2DGS w/ OpacLoss" "2dgss_g9833_oquad1-1000_swc08=2DG-SS" \
    --scenes room ficus mic \
    --val_nums 3 11 22 \
    --val_ordering original \
    --row_labels "bicycle,1=Bicycle" "garden,3=Garden" "kitchen,21=Kitchen" "room,3=Room" \
    "ficus,11=Ficus" "mic,22=Mic" "ship,44=Ship" \
    --row_models "room,3,gt,2dgs_sfm,2dgs_sfm_oquad1-1000,2dgss_g9833_sfm_oquad1-1000_swc08" \
    --inset "bicycle,1,*,475,150,725,400,right,1.0" \
    --inset "garden,3,*,525,400,825,800,right,1.0" \
    --inset "kitchen,21,*,0,0,200,300,left,1.0" \
    --inset "kitchen,21,*,400,0,600,300,right,1.0" \
    --inset "room,3,*,550,0,770,250,right,1.0" \
    --zoom "ficus,11,*,150,0,650,700" \
    --inset "ficus,11,*,300,500,450,700,left,1.0" \
    --inset "ficus,11,*,400,50,550,250,right,1.0" \
    --zoom "mic,22,*,175,0,750,800" \
    --inset "mic,22,*,300,50,400,250,left,1.0" \
    --inset "mic,22,*,200,350,700,800,right,1.0" \
    --zoom "ship,44,*,50,125,775,800" \
    --inset "ship,44,*,300,200,600,600,right,1.0" \
    --output ../results/images/2dgs_slides1.png
