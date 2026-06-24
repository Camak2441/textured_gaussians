python plot_metric_diff.py \
    --models 2dgs_sfm 2dgs_sfm_oquad1-1000 2dgss_g9833_sfm_oquad1-1000_swc08 \
    --model-names 2dgs_sfm=2DGS \
    '2dgs_sfm_oquad1-1000=2DGS$\mathcal{L}_\alpha$' \
    '2dgss_g9833_sfm_oquad1-1000_swc08=2DGSS' \
    --datasets mn360 ns \
    --mode scenes \
    --groups "Indoor:Bonsai,Counter,Kitchen,Room" \
    "Outdoor:Bicycle,Flowers,Garden,Stump,Treehill" \
    "Mip-Nerf 360:Bicycle,Flowers,Garden,Stump,Treehill,Bonsai,Counter,Kitchen,Room" \
    "NeRF-Synthetic:Chair,Drums,Ficus,Hotdog,Lego,Materials,Mic,Ship" \
    --model-overrides \
    "2dgs_sfm:ns=2dgs_g2000" \
    "2dgs_sfm_oquad1-1000:ns=2dgs_g2000_oquad1-1000" \
    "2dgss_g9833_sfm_oquad1-1000_swc08:ns=2dgss_g1966_oquad1-1000_swc08" \
    --metrics psnr \
    --fig-width 8 \
    --fig-height 4 \
    --output ../results/plots/s_per.png
    