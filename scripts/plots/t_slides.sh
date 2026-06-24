python plot_metric_diff.py \
    --models tgs_psfm tgs_b2_psfm \
    --model-names tgs_psfm=TGS 'tgs_b2_psfm=TGS$\partial$' \
    --datasets mn360 ns \
    --mode groups \
    --groups "Indoor:Bonsai,Counter,Kitchen,Room" \
    "Outdoor:Bicycle,Flowers,Garden,Stump,Treehill" \
    "Mip-Nerf 360:Bicycle,Flowers,Garden,Stump,Treehill,Bonsai,Counter,Kitchen,Room" \
    "NeRF-Synthetic:Chair,Drums,Ficus,Hotdog,Lego,Materials,Mic,Ship" \
    --model-overrides \
    "tgs_psfm:NeRF-Synthetic=tgs_g2000" \
    "tgs_b2_psfm:NeRF-Synthetic=tgs_b2_g2000" \
    --metrics psnr \
    --fig-width 3.5 \
    --fig-height 7 \
    --output ../results/plots/b2_psnr.png
python plot_metric_diff.py \
    --models tgs_psfm tgs_b2_psfm \
    --model-names tgs_psfm=TGS 'tgs_b2_psfm=TGS$\partial$' \
    --datasets mn360 ns \
    --mode groups \
    --groups "Indoor:Bonsai,Counter,Kitchen,Room" \
    "Outdoor:Bicycle,Flowers,Garden,Stump,Treehill" \
    "Mip-Nerf 360:Bicycle,Flowers,Garden,Stump,Treehill,Bonsai,Counter,Kitchen,Room" \
    "NeRF-Synthetic:Chair,Drums,Ficus,Hotdog,Lego,Materials,Mic,Ship" \
    --model-overrides "tgs_psfm:NeRF-Synthetic=tgs_g2000" "tgs_b2_psfm:NeRF-Synthetic=tgs_b2_g2000" \
    --metrics ssim \
    --fig-width 3.5 \
    --fig-height 7 \
    --output ../results/plots/b2_ssim.png
python plot_metric_diff.py \
    --models tgs_psfm tgs_b2_psfm \
    --model-names tgs_psfm=TGS 'tgs_b2_psfm=TGS$\partial$' \
    --datasets mn360 ns \
    --mode groups \
    --groups "Indoor:Bonsai,Counter,Kitchen,Room" \
    "Outdoor:Bicycle,Flowers,Garden,Stump,Treehill" \
    "Mip-Nerf 360:Bicycle,Flowers,Garden,Stump,Treehill,Bonsai,Counter,Kitchen,Room" \
    "NeRF-Synthetic:Chair,Drums,Ficus,Hotdog,Lego,Materials,Mic,Ship" \
    --model-overrides "tgs_psfm:NeRF-Synthetic=tgs_g2000" "tgs_b2_psfm:NeRF-Synthetic=tgs_b2_g2000" \
    --metrics lpips \
    --fig-width 3.5 \
    --fig-height 7 \
    --output ../results/plots/b2_lpips.png
python plot_metric_diff.py \
    --models tgs_psfm tgs_b2_psfm \
    --model-names tgs_psfm=TGS 'tgs_b2_psfm=TGS$\partial$' \
    --datasets mn360 ns \
    --mode groups \
    --groups "Indoor:Bonsai,Counter,Kitchen,Room" \
    "Outdoor:Bicycle,Flowers,Garden,Stump,Treehill" \
    "Mip-Nerf 360:Bicycle,Flowers,Garden,Stump,Treehill,Bonsai,Counter,Kitchen,Room" \
    "NeRF-Synthetic:Chair,Drums,Ficus,Hotdog,Lego,Materials,Mic,Ship" \
    --model-overrides \
    "tgs_psfm:NeRF-Synthetic=tgs_g2000" \
    "tgs_b2_psfm:NeRF-Synthetic=tgs_b2_g2000" \
    --metrics cvvdp \
    --fig-width 3.5 \
    --fig-height 7 \
    --output ../results/plots/b2_cvvdp.png
python plot_metric_diff.py \
    --models tgs_psfm tgs_b2_psfm \
    --model-names tgs_psfm=TGS 'tgs_b2_psfm=TGS$\partial$' \
    --datasets mn360 ns \
    --mode groups \
    --groups "Indoor:Bonsai,Counter,Kitchen,Room" \
    "Outdoor:Bicycle,Flowers,Garden,Stump,Treehill" \
    "Mip-Nerf 360:Bicycle,Flowers,Garden,Stump,Treehill,Bonsai,Counter,Kitchen,Room" \
    "NeRF-Synthetic:Chair,Drums,Ficus,Hotdog,Lego,Materials,Mic,Ship" \
    --model-overrides \
    "tgs_psfm:NeRF-Synthetic=tgs_g2000" \
    "tgs_b2_psfm:NeRF-Synthetic=tgs_b2_g2000" \
    --metrics psnr ssim lpips cvvdp \
    --fig-width 2 \
    --fig-height 4 \
    --output ../results/plots/b2.png
    