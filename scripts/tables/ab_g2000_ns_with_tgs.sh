python compile_latex_table.py \
    --font-size footnotesize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics psnr ssim lpips cvvdp \
    --metric-sep 1em \
    --value-sep 0.2em \
    --models tgs_g2000_to0_tgs_b2_g2000_abp mip_tgs_g2000_to0_tgs_b2_g2000_abp_with_tgs aniso_bilinear_tgs_g2000_to0_tgs_b2_g2000_abp_with_tgs \
    --smaller-models 1 \
    --datasets ns
    