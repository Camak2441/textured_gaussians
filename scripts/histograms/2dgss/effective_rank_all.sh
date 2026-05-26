python param_histogram.py \
    --param scale_effective_rank_2d \
    --groups "2dgs_oquad1-1000,mn360,2dgs_g2000_oquad1-1000,ns=2DGS w/ OpacLoss" \
    "2dgss_g9833_oquad1-1000_swc08,mn360,2dgss_g1966_oquad1-1000_swc08,ns=2DG-SS w/ OpacLoss" \
    --weight scene \
    --output ../results/histograms/2dgss_effective_rank_all
