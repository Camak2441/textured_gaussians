python compile_latex_table.py \
    --font-size footnotesize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics psnr ssim lpips cvvdp \
    --metric-sep 1em \
    --models tgs_g2000 tgs_b2_g2000 tgss4_b2_g1999_ot01-0_ott03-0_sgc02_swc08_po_pswc08 \
    --model-params tex_grad \
    --smaller-models \
    --datasets ns
    