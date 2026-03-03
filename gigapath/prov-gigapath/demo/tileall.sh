#!/bin/bash
#SBATCH --job-name=tileall
#SBATCH --partition=ckpt
#SBATCH --account=kurtlab
#SBATCH --array=0-39
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --chdir=/gscratch/kurtlab/models/gigapath/prov-gigapath
#SBATCH --output=logs/tiling/stdout/%A/tiling-%A_%a.out
#SBATCH --error=logs/tiling/stdout/%A/tiling-%A_%a.err


echo "=========================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "Node          : $(hostname)"
echo "Working dir   : $(pwd)"
echo "=========================================="


source ~/.bashrc
conda activate gigapath


export PYTHONUNBUFFERED=1
python3 -m demo.tileall \
  --logdir /gscratch/kurtlab/models/gigapath/prov-gigapath/logs/tiling \
  --num_splits 40 \
  --split_no ${SLURM_ARRAY_TASK_ID}  \

echo "Array task ${SLURM_ARRAY_TASK_ID} finished."