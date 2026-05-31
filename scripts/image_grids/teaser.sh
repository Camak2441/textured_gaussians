#!/usr/bin/bash

python image_grid.py \
    --dpi 150 --cell_gap 0 --font_size 9 --inset_rect \
    --fig_width 100 \
    --border_lw 0.6 --rect_lw 0.4 \
    --models gt tgs_psfm tgss4_b2_g9999_sgc02_swc08_psfm_po_pswc08 \
    --col_labels "gt=Ground Truth" "tgs_psfm=TGS (Base)" "tgss4_b2_g9999_sgc02_swc08_psfm_po_pswc08=TG-SS (Ours)" \
    --scenes chair mic \
    --val_nums 12 3 \
    --val_ordering original \
    --row_labels \
    "chair,12=Chair" "mic,3=Mic" \
    --row_models "chair,12,gt,tgs_g2000,tgss4_b2_g1999_sgc02_swc08_po_pswc08" \
    --row_models "mic,3,gt,tgs_g2000,tgss4_b2_g1999_sgc02_swc08_po_pswc08" \
    --zoom "chair,12,*,150,0,650,750" \
    --zoom "mic,3,*,0,0,650,700" \
    --output ../assets/teaser.png
