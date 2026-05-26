python compile_latex_table.py \
    --font-size footnotesize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics render_time memory model_size \
    --metric-sep 1em \
    --value-sep 0.2em \
    --models tgs_g2000_to0_tgs_b2_g2000_abp mip_tgs_g2000_to0_tgs_b2_g2000_abp aniso_bilinear_tgs_g2000_to0_tgs_b2_g2000_abp \
    --model-names \
    tgs_g2000_to0_tgs_b2_g2000_abp=Bilin \
    mip_tgs_g2000_to0_tgs_b2_g2000_abp=Mip \
    aniso_bilinear_tgs_g2000_to0_tgs_b2_g2000_abp=Aniso \
    --smaller-models 1 \
    --datasets nsa
    