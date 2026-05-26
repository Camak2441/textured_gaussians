python param_histogram.py \
    --param scale_area \
    --groups "2dgs,mn360,2dgs_g2000,ns=2DGS" \
    "2dgss_g9833_oquad1-1000_swc08,mn360,2dgss_g1966_oquad1-1000_swc08,ns=2DG-SS" \
    --weight scene \
    --percentile None 70 \
    --output ../results/histograms/2dgss_area_all
