python compile_latex_table.py \
    --font-size scriptsize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics render_time memory model_size \
    --metric-sep 1em \
    --models 2dgs 2dgss_g9833_oquad1-1000_swc08 2dgs_sfm 2dgss_g9833_sfm_oquad1-1000_swc08 \
    --model-names 2dgs=GS 2dgss_g9833_oquad1-1000_swc08=G-SS 2dgs_sfm=GS 2dgss_g9833_sfm_oquad1-1000_swc08=G-SS \
    --group-by init \
    --datasets mn360
    