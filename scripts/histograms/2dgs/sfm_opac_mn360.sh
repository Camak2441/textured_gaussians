python param_histogram.py \
    --param opacities_activated \
    --models 2dgs_sfm=2DGS "2dgs_sfm_oquad1-1000=2DGS w/ OpacLoss" \
    --datasets mn360 \
    --weight scene \
    --output ../results/histograms/2dgs_sfm_mn360
