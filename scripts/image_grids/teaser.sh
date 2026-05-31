#!/usr/bin/bash

python image_grid.py \
    --dpi 900 --cell_gap 0 --font_size 9 --inset_rect \
    --border_lw 0.6 --rect_lw 0.4 \
    --models gt tgs_psfm tgss4_b2_g9999_sgc02_swc08_psfm_po_pswc08 \
    --col_labels "gt=Ground Truth" "tgs_psfm=TGS (Base)" "tgss4_b2_g9999_sgc02_swc08_psfm_po_pswc08=TG-SS (Ours)" \
    --scenes bonsai garden chair mic \
    --val_nums 5 1 12 3 \
    --val_ordering original \
    --row_labels "bonsai,5=Bonsai" "garden,1=Garden" \
    "chair,12=Chair" "mic,3=Mic" \
    --row_models "chair,12,gt,tgs_g2000,tgss4_b2_g1999_sgc02_swc08_po_pswc08" \
    --row_models "mic,3,gt,tgs_g2000,tgss4_b2_g1999_sgc02_swc08_po_pswc08" \
    --zoom "counter,9,*,0,0,500,1000" \
    --zoom "garden,1,*,250,0,1100,750" \
    --zoom "room,29,*,150,0,2000,750" \
    --zoom "chair,12,*,150,0,650,750" \
    --zoom "drums,3,*,50,50,750,750" \
    --zoom "lego,5,*,25,100,750,650" \
    --zoom "mic,3,*,0,0,650,700" \
    --zoom "ship,5,*,100,100,725,650" \
    --output ../assets/teaser.png
