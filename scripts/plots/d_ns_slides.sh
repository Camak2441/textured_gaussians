python plot_metric_diff.py \
    --models tgs_b2_ta_t6 dtgs3_b2_ta_t8 \
    --model-names tgs_b2_ta_t6=TGS dtgs3_b2_ta_t8=DTGS \
    --datasets ns \
    --mode groups \
    --groups "NeRF-Synthetic:Chair,Drums,Ficus,Hotdog,Lego,Materials,Mic,Ship" \
    --model-overrides \
    "tgs_b2_ta_t6:NeRF-Synthetic=tgs_g2000" \
    "dtgs3_b2_ta_t8:NeRF-Synthetic=tgs_b2_g2000" \
    --metrics psnr ssim lpips cvvdp \
    --fig-width 2 \
    --fig-height 4 \
    --output ../results/plots/d.png
    