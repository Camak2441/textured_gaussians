python param_histogram.py \
    --param steepnesses_activated \
    --groups "2dss_g2000_k2,chair=2D-SS k2" \
    "2dss_g2000_k3,chair=2D-SS k3" \
    "2dss_g2000_k5,chair=2D-SS k5" \
    "2dss_g2000_k10,chair=2D-SS k10" \
    --weight scene \
    --output ../results/histograms/2dss_sharpness_all
