python compile_latex_table.py \
    --font-size scriptsize \
    --cell-highlights "color_hla" "color_hlb" \
    --metrics psnr ssim lpips cvvdp \
    --metric-sep 1em \
    --value-sep 0.2em \
    --models 2dgs_g2000 2dgs_g2000_oquad1-1000 2dgss_g1966_oquad1-1000_swc08 \
    2dgs 2dgs_oquad1-1000 2dgss_g9833_oquad1-1000_swc08 \
    --model-names 2dgs_g2000=GS '2dgs_g2000_oquad1-1000=GS$\lopac$' 2dgss_g1966_oquad1-1000_swc08=G-SS \
    2dgs=GS '2dgs_oquad1-1000=GS$\lopac$' 2dgss_g9833_oquad1-1000_swc08=G-SS \
    --groups 'splats:(2000,1966)=$\sim$2k:(10000,9833)=$\sim$10k' \
    --datasets ns
