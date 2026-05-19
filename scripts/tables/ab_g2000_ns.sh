python compile_latex_table.py \
    --font-size footnotesize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics psnr ssim lpips cvvdp \
    --metric-sep 1em \
    --models tgs_g2000_to0_tgs_b2_g2000_abp aniso_bilinear_tgs_g2000_to0_tgs_b2_g2000_abp \
    --smaller-models 1 \
    --datasets ns
    