# MEDIQA Example Submission

## Directory Contents
The example submission consists of three components: 
1) Histopathology-specific linear probe (`corebt_histo_main.py`)
2) MRI-specific linear probe (`corebt_mri_main.py`)
3) Linear probe fusion with adapter steps (`corebt_fusion_main.py`)

All scripts can be run together from `run_corebt_linear_probes.py`

## Setup

```bash
# Install uv 
curl -LsSf https://astral.sh/uv/install.sh | sh 

# create env by installing packages from uv.lock file, activate env
cd CoreBT/experiments_mediqa
uv sync

# verify the installation and CUDA support
uv run python -c "import torch; print(f'Torch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')"
```

## Prerequisites
Download the embeddings files, and create the dataset CSVs with the subject_id and the label headers. For examples look at `dataset_csvs`. 

> Note that within `dataset_csvs`, only `train.csv` contains correct labels, while the rest contain random labels.

The minimal dataset csv should be structured like this:
<div align="center">

| subject_id | level1_label | lgghgg_label | who_grade_label |
| :--- | :--- | :--- | :--- |
| CoReBT-0037 | 3 | 0 | 0 |
| CoReBT-0038 | 2 | 1 | 1 |
| ... | ... | ... | ... |
</div>


## Example commands
> Note: Because K-fold CV is not implemented yet, the following scripts use the `train.csv` for both training and validation. 
The code below will perform inference on the subjects in `TEST_CSV` and `TEST_MRI/TEST_HISTO`, so we pass in the validation subjects there to generate the predictions for those specific samples.

### Shared variables

```bash
CSV_DIR=/gscratch/kurtlab/CoreBT/experiments_mediqa/dataset_csvs
EMBED_DIR=/gscratch/scrubbed/juampablo/corebt_dataset
OUTPUT_DIR=/gscratch/kurtlab/CoreBT/experiments_mediqa/runs

# Base Directories 
CSV_DIR="dataset_csvs"
EMBED_DIR="/gscratch/scrubbed/juampablo/corebt_dataset"

# Metadata CSV Paths 
TRAIN_CSV="$CSV_DIR/train.csv"
VAL_CSV="$CSV_DIR/train.csv"          
TEST_CSV="$CSV_DIR/val_randomized.csv"

# MRI Embedding Paths
TRAIN_MRI="$EMBED_DIR/MRI_Embeddings_train.zip"
VAL_MRI="$EMBED_DIR/MRI_Embeddings_train.zip"
TEST_MRI="$EMBED_DIR/MRI_Embeddings_val.zip"

# Histopathology Embedding Paths
TRAIN_HISTO="$EMBED_DIR/Pathology_Embeddings_train.zip"
VAL_HISTO="$EMBED_DIR/Pathology_Embeddings_train.zip"
TEST_HISTO="$EMBED_DIR/Pathology_Embeddings_val.zip"

# Hyperparameters 
OUTPUT_DIR="runs"
MRI_EMBED_DIM=768
HISTO_EMBED_DIM=768
BATCH_SIZE=32
TRAIN_ITERS=600
LR=0.001
MIN_LR=0.0
OPTIM="adam"
MOMENTUM=0.0
WEIGHT_DECAY=1e-4
EVAL_INTERVAL=10
NUM_WORKERS=4
SEED=42
LABEL_PREFIX="all" # choices: level1, lgghgg, who_grade, all

```


### All models training with `run_corebt_linear_probes.py`
```bash
uv run python run_corebt_linear_probes.py \
    --train_csv_path "$TRAIN_CSV" \
    --val_csv_path "$VAL_CSV" \
    --test_csv_path "$TEST_CSV" \
    --train_mri_embed_path "$TRAIN_MRI" \
    --val_mri_embed_path "$VAL_MRI" \
    --test_mri_embed_path "$TEST_MRI" \
    --train_histo_embed_path "$TRAIN_HISTO" \
    --val_histo_embed_path "$VAL_HISTO" \
    --test_histo_embed_path "$TEST_HISTO" \
    --output_dir "$OUTPUT_DIR" \
    --label_prefix "$LABEL_PREFIX" \
    --variant "all" # choices=["mri", "histo", "fusion", "all"], default="all"
```

### MRI linear probe training with `corebt_mri_main.py`

```bash
uv run corebt_mri_main.py \
    --train_csv_path "$TRAIN_CSV" \
    --val_csv_path "$VAL_CSV" \
    --test_csv_path "$TEST_CSV" \
    --train_mri_embed_path "$TRAIN_MRI" \
    --val_mri_embed_path "$VAL_MRI" \
    --test_mri_embed_path "$TEST_MRI" \
    --label_prefix $LABEL_PREFIX \
    --embed_dim $MRI_EMBED_DIM \
    --batch_size $BATCH_SIZE \
    --train_iters $TRAIN_ITERS \
    --lr $LR \
    --min_lr $MIN_LR \
    --optim $OPTIM \
    --momentum $MOMENTUM \
    --weight_decay $WEIGHT_DECAY \
    --eval_interval $EVAL_INTERVAL \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR/mri"

```

### Histopathology linear probe training with `corebt_histo_main.py`
```bash
uv run corebt_histo_main.py \
    --train_csv_path "$TRAIN_CSV" \
    --val_csv_path "$VAL_CSV" \
    --test_csv_path "$TEST_CSV" \
    --train_histo_embed_path "$TRAIN_HISTO" \
    --val_histo_embed_path "$VAL_HISTO" \
    --test_histo_embed_path "$TEST_HISTO" \
    --label_prefix $LABEL_PREFIX \
    --embed_dim $HISTO_EMBED_DIM \
    --batch_size $BATCH_SIZE \
    --train_iters $TRAIN_ITERS \
    --lr $LR \
    --min_lr $MIN_LR \
    --optim $OPTIM \
    --momentum $MOMENTUM \
    --weight_decay $WEIGHT_DECAY \
    --eval_interval $EVAL_INTERVAL \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR/histopathology"
```


### MRI + Histopathology Fusion adapter training with `corebt_histo_main.py`

```bash
uv run corebt_fusion_main.py \
    --train_csv_path "$TRAIN_CSV" \
    --val_csv_path "$VAL_CSV" \
    --test_csv_path "$TEST_CSV" \
    --train_mri_embed_path "$TRAIN_MRI" \
    --val_mri_embed_path "$VAL_MRI" \
    --test_mri_embed_path "$TEST_MRI" \
    --train_histo_embed_path "$TRAIN_HISTO" \
    --val_histo_embed_path "$VAL_HISTO" \
    --test_histo_embed_path "$TEST_HISTO" \
    --label_prefix $LABEL_PREFIX \
    --histo_embed_dim $HISTO_EMBED_DIM \
    --mri_embed_dim $MRI_EMBED_DIM \
    --batch_size $BATCH_SIZE \
    --train_iters $TRAIN_ITERS \
    --lr $LR \
    --min_lr $MIN_LR \
    --optim $OPTIM \
    --momentum $MOMENTUM \
    --weight_decay $WEIGHT_DECAY \
    --eval_interval $EVAL_INTERVAL \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR/fusion"
```


## Collecting predictions
Our submitted predictions are made using a combination of the predictions from the MRI probe and the Histopathology probe. The `utils/collect_final_prediction.py` file combines the labels, taking the MRI labels as priority. Modify this with your own paths if using.

```bash
uv run utils/collect_final_prediction.py
```

This saves the prediction file to `run/prediction.csv`.

## Submitting predictions
Submissions are made as zip, with only the `prediction.csv` file inside.
```bash
zip prediction-val.zip prediction.csv
```