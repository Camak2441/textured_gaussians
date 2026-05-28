python compile_latex_table.py \
    --font-size footnotesize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics render_time memory model_size \
    --metric-sep 1em \
    --value-sep 0.2em \
    --models tgs_g2000 tgs_b2_g2000 tgs_b2_g2000_poquad1 tgss4_b2_g1999_sgc02_swc08_po_pswc08 \
    --model-names tgs_g2000=GS 'tgs_b2_g2000=GS$\partial$' 'tgs_b2_g2000_poquad1=GS$\lopac$' 'tgss4_b2_g1999_sgc02_swc08_po_pswc08=G-SS' \
    --model-params tex_grad \
    --smaller-models \
    --datasets nsa
    