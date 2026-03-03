#!/bin/bash


python3 run_corebt_linear_probes.py \
  --dataset_csv /gscratch/kurtlab/CoreBT/dataset_utils/fusion/corebt_fusion_dataset_WHOGrade_case_sharedtest.csv \
  --run_name WHOGrade 
 
python3 run_corebt_linear_probes.py \
  --dataset_csv /gscratch/kurtlab/CoreBT/dataset_utils/fusion/corebt_fusion_dataset_LGG_HGG_case_sharedtest.csv \
  --run_name LGG_HGG  

python3 run_corebt_linear_probes.py \
  --dataset_csv /gscratch/kurtlab/CoreBT/dataset_utils/fusion/corebt_fusion_dataset_LEVEL1_case_sharedtest.csv \
  --run_name Level1Class 