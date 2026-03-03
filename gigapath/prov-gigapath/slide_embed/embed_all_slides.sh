#!/bin/bash
#SBATCH --job-name=slide_embed_all
#SBATCH --partition=ckpt
#SBATCH --account=kurtlab
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --chdir=/gscratch/kurtlab/models/gigapath/prov-gigapath
#SBATCH --output=logs/slide_embed/stdout/%A/slide_embed-%A_%a.out
#SBATCH --error=logs/slide_embed/stderr/%A/slide_embed-%A_%a.err


echo "=========================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "Node          : $(hostname)"
echo "Working dir   : $(pwd)"
echo "=========================================="


source ~/.bashrc
conda activate gigapath
module load gcc/11

export PYTHONUNBUFFERED=1
TILE_EMBEDDINGS_DIR=/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/tile_embeddings/h5_files
SLIDE_EMBEDDINGS_DIR=/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/slide_embeddings
NUM_SPLITS=4
SPLIT_NO=${SLURM_ARRAY_TASK_ID}


python3 -m slide_embed.embed_all_slides --tile_embeddings_dir $TILE_EMBEDDINGS_DIR --slide_embeddings_dir $SLIDE_EMBEDDINGS_DIR --num_splits $NUM_SPLITS --split_no $SPLIT_NO
