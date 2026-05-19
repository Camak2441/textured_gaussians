python compile_latex_table.py \
    --font-size scriptsize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics render_time memory model_size \
    --metric-sep 1em \
    --models 2dgs_g2000 2dgss_g1966_oquad1-1000_swc08 2dgs 2dgss_g9833_oquad1-1000_swc08 \
    --groups 'splats:(2000,1966)=$\sim$2k:(10000,9833)=$\sim$10k' \
    --model-names 2dgs_g2000=GS 2dgss_g1966_oquad1-1000_swc08=G-SS 2dgs=GS 2dgss_g9833_oquad1-1000_swc08=G-SS \
    --datasets ns
