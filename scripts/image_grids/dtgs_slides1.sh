#!/usr/bin/bash

python image_grid.py \
    --dpi 900 --cell_gap 0 --font_size 9 --inset_rect \
    --border_lw 0.6 --rect_lw 0.4 \
    --models gt tgs_b2_ta_t6 dtgs3_b2_ta_t8 \
    --col_labels "gt=Ground Truth" "tgs_b2_ta_t6=TGS" "dtgs3_b2_ta_t8=DTGS" \
    --scenes ficus lego \
    --val_ordering original \
    --val_nums 59 6 5 \
    --row_labels "ficus,59=Ficus" "lego,6=Lego" \
    --zoom "chair,99,*,150,50,625,750" \
    --inset "chair,99,*,550,100,600,300,right,1.0" \
    --zoom "ficus,59,*,150,0,600,725" \
    --inset "ficus,59,*,250,175,350,425,left,1.0" \
    --inset "ficus,59,*,400,50,500,300,right,1.0" \
    --zoom "lego,6,*,25,100,700,625" \
    --inset "lego,6,*,200,200,300,425,left,1.0" \
    --inset "lego,6,*,400,175,500,350,right,1.0" \
    --zoom "mic,5,*,25,100,650,700" \
    --inset "mic,5,*,100,400,250,625,left,1.0" \
    --inset "mic,5,*,500,100,600,300,right,1.0" \
    --output ../results/images/dtgs_slides1.png
    