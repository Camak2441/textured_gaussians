bash train/run_mip_nerf_360.sh "transfer/ckpts_29999.sh $1"
bash train/run_nerf_synthetic.sh "transfer/ckpts_29999.sh $1"
