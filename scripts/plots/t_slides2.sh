python plot_metric_diff.py \
    --models tgs_psfm tgs_b2_psfm \
    --model-names tgs_psfm=TGS 'tgs_b2_psfm=TGS$\partial$' \
    --datasets mn360 ns \
    --mode scenes \
    --groups "Indoor:Bonsai,Counter,Kitchen,Room" \
    "Outdoor:Bicycle,Flowers,Garden,Stump,Treehill" \
    "Mip-Nerf 360:Bicycle,Flowers,Garden,Stump,Treehill,Bonsai,Counter,Kitchen,Room" \
    "NeRF-Synthetic:Chair,Drums,Ficus,Hotdog,Lego,Materials,Mic,Ship" \
    --model-overrides \
    "tgs_psfm:ns=tgs_g2000" "tgs_b2_psfm:ns=tgs_b2_g2000" \
    --metrics psnr \
    --fig-width 8 \
    --fig-height 4 \
    --output ../results/plots/b2_per.png
    