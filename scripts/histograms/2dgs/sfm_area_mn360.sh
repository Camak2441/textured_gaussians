python param_histogram.py \
    --param scale_area \
    --models 2dgs=2DGS "2dgs_sfm=2DGS w/ SfM" \
    --datasets mn360 \
    --weight scene \
    --percentile None 90 \
    --output ../results/histograms/2dgs_sfm_area_mn360
