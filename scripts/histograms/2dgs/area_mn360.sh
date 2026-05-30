python param_histogram.py \
    --param scale_area \
    --groups "2dgs,mn360=2DGS" \
    "2dgs_oquad1-1000,mn360=2DGS w/ OpacLoss" \
    --weight scene \
    --percentile None 80 \
    --title "Histogram of Splat Area on Mip-Nerf 360 with Random Initialisation" \
    --output ../results/histograms/2dgs_area_mn360
