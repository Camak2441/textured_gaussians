python param_histogram.py \
    --param opacities_activated \
    --groups "2dgs_oquad1-1000,mn360,2dgs_g2000_oquad1-1000,ns=2DGS w/ OpacLoss" \
    "2dgss_g9833_oquad1-1000_swc08,mn360,2dgss_g1966_oquad1-1000_swc08,ns=2DG-SS w/ OpacLoss" \
    --weight scene \
    --title "Histogram of Opacity on Nerf-Synthetic and Mip-Nerf 360 with Random Initialisation" \
    --output ../results/histograms/2dgss_opac_all
