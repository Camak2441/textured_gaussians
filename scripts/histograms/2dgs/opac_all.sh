python param_histogram.py \
    --param opacities_activated \
    --groups "2dgs,mn360,2dgs_g2000,ns=2DGS" \
    "2dgs_oquad1-1000,mn360,2dgs_g2000_oquad1-1000,ns=2DGS w/ OpacLoss" \
    --weight scene \
    --title "Histogram of Opacity on Nerf-Synthetic and Mip-Nerf 360 with Random Initialisation" \
    --output ../results/histograms/2dgs_opac_all
