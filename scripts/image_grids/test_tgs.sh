#!/usr/bin/bash

python image_grid.py \
    --dpi 600 --cell_gap 0 --font_size 9 --inset_rect \
    --models gt tgs_psfm tgs_b2_psfm tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_psfm_po_pswc08 \
    --col_labels "gt=Ground Truth" "tgs_psfm=TGS" "tgs_b2_psfm=TGS w/ TexGrad" "tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_psfm_po_pswc08=TGSS" \
    --scenes mic \
    --val_nums 3 \
    --row_models "mic,3,gt,tgs_g2000,tgs_b2_g2000,tgss4_b2_g1999_ot01-0_ott03-0_sgc02_swc08_po_pswc08" \
    --row_models "ship,5,gt,tgs_g2000,tgs_b2_g2000,tgss4_b2_g1999_ot01-0_ott03-0_sgc02_swc08_po_pswc08" \
    --zoom "room,29,*,150,0,2000,750" \
    --inset "room,29,*,650,100,750,300,right,1.0" \
    --output ../results/images/test_tgs.png

#   --zoom "counter,9,*,0,150,500,550" \