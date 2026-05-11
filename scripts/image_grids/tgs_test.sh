#!/usr/bin/bash

python image_grid.py \
    --dpi 600 --cell_gap 0 --font_size 9 --inset_rect \
    --models gt tgs_psfm tgs_b2_psfm tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_psfm_po_pswc08 \
    --col_labels "gt=Ground Truth" "tgs_psfm=TGS" "tgs_b2_psfm=TGS w/ TexGrad" "tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_psfm_po_pswc08=TGSS" \
    --scenes lego \
    --val_nums 5 \
    --row_models "lego,5,gt,tgs_g2000,tgs_b2_g2000,tgss4_b2_g1999_ot01-0_ott03-0_sgc02_swc08_po_pswc08" \
    --zoom "lego,5,*,25,100,750,650" \
    --inset "lego,5,*,100,250,200,550,left,1.0" \
    --inset "lego,5,*,300,175,450,525,right,1.0" \
    --output ../results/images/tgs_test.png
