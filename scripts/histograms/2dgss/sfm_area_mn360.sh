python param_histogram.py \
    --param scale_area \
    --models 2dgs_oquad1-1000=2DGS "2dgs_sfm_oquad1-1000=2DGS w/ SfM" \
    "2dgss_g9833_oquad1-1000_swc08=2DG-SS" "2dgss_g9833_sfm_oquad1-1000_swc08=2DG-SS w/ SfM" \
    --datasets mn360 \
    --weight scene \
    --percentile None 50 \
    --output ../results/histograms/2dgss_sfm_area_mn360
