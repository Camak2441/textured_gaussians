python compile_latex_table.py \
    --font-size footnotesize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics render_time memory model_size \
    --metric-sep 1em \
    --value-sep 0.2em \
    --models tgs_to0_tgs_b2_psfm_abp mip_tgs_to0_tgs_b2_psfm_abp aniso_bilinear_tgs_to0_tgs_b2_psfm_abp \
    --model-names \
    tgs_to0_tgs_b2_psfm_abp=Bilin \
    mip_tgs_to0_tgs_b2_psfm_abp=Mip \
    aniso_bilinear_tgs_to0_tgs_b2_psfm_abp=Aniso \
    --model-params tex_grad \
    --smaller-models 1 \
    --datasets mn360a
    