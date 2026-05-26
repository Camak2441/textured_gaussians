python param_histogram.py \
    --param means_displacement \
    --groups "tgs_g2000,ns=TGS" \
    "tgs_b2_g2000,ns=TGS w/ TexGrad" \
    --weight scene \
    --percentile None 80 \
    --output ../results/histograms/tgs_displacement_ns
