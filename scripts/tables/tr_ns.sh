python compile_latex_table.py \
    --font-size footnotesize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics render_time memory model_size \
    --metric-sep 1em \
    --models tgs tgs_b2 tgss4_b2_g9999_ot01-0_ott03-0_sgc02_swc08_po_pswc08 \
    --model-params tex_grad \
    --smaller-models \
    --datasets ns
    
    