#!/bin/bash
#SBATCH --job-name=tile_embed_all
#SBATCH --partition=ckpt
#SBATCH --account=kurtlab
#SBATCH --array=0-19
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --chdir=/gscratch/kurtlab/models/gigapath/prov-gigapath
#SBATCH --output=logs/tile_embed/stdout/%A/tile_embed-%A_%a.out
#SBATCH --error=logs/tile_embed/stderr/%A/tile_embed-%A_%a.err


echo "=========================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "Node          : $(hostname)"
echo "Working dir   : $(pwd)"
echo "=========================================="


source ~/.bashrc
conda activate gigapath
export PYTHONUNBUFFERED=1


DATA_H5_DIR=/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/corebt_tiles/patches
DATA_SLIDE_DIR=/gscratch/scrubbed/juampablo/corebt/corebt_pathology
CSV_PATH=/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/corebt_tiles/process_list_autogen.csv
FEAT_DIR=/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/tile_embeddings
NUM_SPLITS=20
SPLIT_NO=${SLURM_ARRAY_TASK_ID}


python3 -m tile_embed.extract_features_gigapath_fp --data_h5_dir $DATA_H5_DIR --data_slide_dir $DATA_SLIDE_DIR --csv_path $CSV_PATH --feat_dir $FEAT_DIR --split_no $SPLIT_NO --num_splits $NUM_SPLITS

