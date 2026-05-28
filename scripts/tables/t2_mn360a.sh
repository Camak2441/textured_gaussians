python compile_latex_table.py \
    --font-size scriptsize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics psnr ssim lpips cvvdp \
    --metric-sep 1em \
    --value-sep 0.2em \
    --models tgs tgs_b2 tgs_b2_poquad1 tgss4_b2_g9999_sgc02_swc08_po_pswc08 \
    --model-params tex_grad opac_loss \
    --model-names tgs=GS 'tgs_b2=GS$\partial$' 'tgs_b2_poquad1=GS$\lopac$' 'tgss4_b2_g9999_sgc02_swc08_po_pswc08=G-SS' \
    --smaller-models \
    --datasets mn360a
    