python param_histogram.py \
    --param means_displacement \
    --groups "tgs,mn360,tgs_g2000,ns=TGS" \
    "tgs_b2,mn360,tgs_b2_g2000,ns=TGS w/ TexGrad" \
    --weight scene \
    --percentile None 50 \
    --title "Histogram of the Splat Displacement on both Mip-Nerf 360 and Nerf-Synthetic" \
    --output ../results/histograms/tgs_displacement_all
