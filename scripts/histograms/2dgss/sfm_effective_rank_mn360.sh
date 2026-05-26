python param_histogram.py \
    --param scale_effective_rank_2d \
    --groups "2dgs_sfm_oquad1-1000,mn360=2DGS w/ OpacLoss" \
    "2dgss_g9833_sfm_oquad1-1000_swc08,mn360=2DG-SS w/ OpacLoss" \
    --weight scene \
    --output ../results/histograms/2dgss_sfm_effective_rank_mn360
