python compile_latex_table.py \
    --font-size footnotesize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics render_time memory model_size \
    --metric-sep 1em \
    --value-sep 0.2em \
    --models tgs_psfm tgs_b2_psfm tgs_b2_psfm_poquad1 tgss4_b2_g9999_sgc02_swc08_psfm_po_pswc08 \
    --model-params tex_grad opac_loss \
    --model-names tgs_psfm=GS 'tgs_b2_psfm=GS$\partial$' 'tgs_b2_psfm_poquad1=GS$\lopac$' 'tgss4_b2_g9999_sgc02_swc08_psfm_po_pswc08=G-SS' \
    --smaller-models \
    --datasets mn360a
    