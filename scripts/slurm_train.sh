#!/bin/bash
#SBATCH -J tgs_train
#SBATCH -A <ACCOUNT>           # e.g. XXXXXXXXX-SL2-GPU
# Container: Python 3.12, PyTorch 2.9.1, CUDA 12.8, target sm_80 (Ampere A100)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH -p ampere
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

module load singularity

SIF=~/containers/tgs.sif
REPO=~/textured-gaussians
RDS=/rds/user/$USER/hpc-work

mkdir -p $REPO/logs

singularity exec --nv \
    --bind $REPO:$REPO \
    --bind $RDS:$RDS \
    $SIF \
    bash -c "
        cd $REPO/examples
        python simple_trainer_textured_gaussians.py mcmc \
            --scene ${SCENE:-chair} \
            --model_type ${MODEL:-2dgs} \
            --init_num_pts 10000 \
            --strategy.cap-max 10000 \
            --texture_resolution 50 \
            --disable_viewer
    "
