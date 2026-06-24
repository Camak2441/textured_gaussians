python plot_metric_diff.py \
    --models tgs_b2_psfm tgs_b2_psfm_poquad1 tgss4_b2_g9999_sgc02_swc08_psfm_po_pswc08 \
    --model-names 'tgs_b2_psfm=TGS$\partial$' \
    'tgs_b2_psfm_poquad1=TGS$\mathcal{L}_\alpha$' \
    'tgss4_b2_g9999_sgc02_swc08_psfm_po_pswc08=TGSS' \
    --datasets mn360 ns \
    --mode groups \
    --groups "Indoor:Bonsai,Counter,Kitchen,Room" \
    "Outdoor:Bicycle,Flowers,Garden,Stump,Treehill" \
    "Mip-Nerf 360:Bicycle,Flowers,Garden,Stump,Treehill,Bonsai,Counter,Kitchen,Room" \
    "NeRF-Synthetic:Chair,Drums,Ficus,Hotdog,Lego,Materials,Mic,Ship" \
    --model-overrides \
    "tgs_b2_psfm:NeRF-Synthetic=tgs_b2_g2000" \
    "tgs_b2_psfm_poquad1:NeRF-Synthetic=tgs_b2_g2000_poquad1" \
    "tgss4_b2_g9999_sgc02_swc08_psfm_po_pswc08:NeRF-Synthetic=tgss4_b2_g1999_sgc02_swc08_po_pswc08" \
    --metrics psnr ssim lpips cvvdp \
    --fig-width 2 \
    --fig-height 4 \
    --output ../results/plots/st.png
    