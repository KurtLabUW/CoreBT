#!/bin/bash
#SBATCH --job-name=tileall
#SBATCH --partition=ckpt
#SBATCH --account=kurtlab
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --chdir=/gscratch/kurtlab/models/CLAM/
#SBATCH --output=logs/tiling/stdout/%A/tiling-%A_%a.out
#SBATCH --error=logs/tiling/stdout/%A/tiling-%A_%a.err


echo "=========================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $(hostname)"
echo "Working dir   : $(pwd)"
echo "=========================================="


source ~/.bashrc
conda activate clam_latest


DATA_DIRECTORY=/gscratch/scrubbed/juampablo/corebt/corebt_pathology
RESULTS_DIRECTORY=/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/corebt_tiles
python3 create_patches_fp.py --source $DATA_DIRECTORY --save_dir $RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch 