#!/usr/bin/bash

python image_grid.py \
    --dpi 900 --cell_gap 0 --font_size 9 --inset_rect \
    --border_lw 0.6 --rect_lw 0.4 \
    --models gt tgs_b2_ta_t6 dtgs3_b2_ta_t8 \
    --col_labels "gt=Ground Truth" "tgs_b2_ta_t6=TGS" "dtgs3_b2_ta_t8=DTGS" \
    --scenes chair ficus lego mic \
    --val_nums 99 59 6 5 \
    --row_labels "mic,5=Mic" \
    --output ../results/images/dtgs.png