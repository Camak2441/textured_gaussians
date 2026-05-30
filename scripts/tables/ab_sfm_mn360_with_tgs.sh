python compile_latex_table.py \
    --font-size footnotesize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics psnr ssim lpips cvvdp \
    --metric-sep 1em \
    --value-sep 0.2em \
    --models tgs_to0_tgs_b2_psfm_abp mip_tgs_to0_tgs_b2_psfm_abp_with_tgs aniso_bilinear_tgs_to0_tgs_b2_psfm_abp_with_tgs \
    --model-names \
    tgs_to0_tgs_b2_psfm_abp=Bilin \
    mip_tgs_to0_tgs_b2_psfm_abp_with_tgs=Mip \
    aniso_bilinear_tgs_to0_tgs_b2_psfm_abp_with_tgs=Aniso \
    --smaller-models 1 \
    --datasets mn360
    