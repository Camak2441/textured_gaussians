python compile_latex_table.py \
    --font-size footnotesize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics psnr ssim lpips cvvdp \
    --metric-sep 1em \
    --models tgs_to0_tgs_b2_psfm_abp \
    --model-params tex_grad \
    --smaller-models 1 \
    --datasets mn360
    