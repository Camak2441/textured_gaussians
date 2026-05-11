#!/usr/bin/bash

python image_grid.py \
    --dpi 900 --cell_gap 0 --font_size 9 --inset_rect \
    --models gt tgs_psfm tgs_b2_psfm tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_psfm_po_pswc08 \
    --col_labels "gt=Ground Truth" "tgs_psfm=TGS" "tgs_b2_psfm=TGS w/ TexGrad" "tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_psfm_po_pswc08=TGSS" \
    --scenes bonsai counter garden room chair drums mic ship \
    --val_nums 5 9 1 29 12 3 3 5 \
    --row_labels "bonsai,5=Bonsai" "counter,9=Counter" "garden,1=Garden" "room,29=Room" \
    "chair,12=Chair" "drums,3=Drums" "mic,3=Mic" "ship,5=Ship" \
    --row_models "chair,12,gt,tgs_g2000,tgs_b2_g2000,tgss4_b2_g1999_ot01-0_ott03-0_sgc02_swc08_po_pswc08" \
    --row_models "drums,3,gt,tgs_g2000,tgs_b2_g2000,tgss4_b2_g1999_ot01-0_ott03-0_sgc02_swc08_po_pswc08" \
    --row_models "mic,3,gt,tgs_g2000,tgs_b2_g2000,tgss4_b2_g1999_ot01-0_ott03-0_sgc02_swc08_po_pswc08" \
    --row_models "ship,5,gt,tgs_g2000,tgs_b2_g2000,tgss4_b2_g1999_ot01-0_ott03-0_sgc02_swc08_po_pswc08" \
    --inset "bonsai,5,*,175,100,300,250,left,1.0" \
    --zoom "counter,9,*,0,0,500,1000" \
    --inset "counter,9,*,75,250,150,390,left,1.0" \
    --inset "counter,9,*,350,360,450,500,right,1.0" \
    --zoom "garden,1,*,250,0,1100,750" \
    --inset "garden,1,*,500,25,700,350,left,1.0" \
    --inset "garden,1,*,500,400,650,700,right,1.0" \
    --zoom "room,29,*,150,0,2000,750" \
    --inset "room,29,*,650,100,750,300,right,1.0" \
    --zoom "chair,12,*,150,0,650,750" \
    --inset "chair,12,*,350,50,450,190,left,1.0" \
    --inset "chair,12,*,350,360,450,500,right,1.0" \
    --zoom "drums,3,*,50,50,750,750" \
    --inset "drums,3,*,150,375,300,525,right,1.0" \
    --output ../results/images/tgs.png

#   --zoom "counter,9,*,0,150,500,550" \