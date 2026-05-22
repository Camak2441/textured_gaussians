python compile_latex_table.py \
    --font-size footnotesize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics render_time memory model_size \
    --metric-sep 1em \
    --models tgs_b2_ta_t6 dtgs3_b2_ta_t8 \
    --smaller-models \
    --datasets ns
