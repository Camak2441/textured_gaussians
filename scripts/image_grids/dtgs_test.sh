#!/usr/bin/bash

python image_grid.py \
    --dpi 900 --cell_gap 0 --font_size 9 --inset_rect \
    --border_lw 0.6 --rect_lw 0.4 \
    --models gt tgs_b2_ta_t6 dtgs3_b2_ta_t8 \
    --col_labels "gt=Ground Truth" "tgs_b2_ta_t6=TGS" "dtgs3_b2_ta_t8=DTGS" \
    --scenes mic \
    --val_nums 5 \
    --zoom "mic,5,*,25,100,650,700" \
    --inset "mic,5,*,100,400,250,625,left,1.0" \
    --inset "mic,5,*,500,100,600,300,right,1.0" \
    --output ../results/images/dtgs_test.png
    