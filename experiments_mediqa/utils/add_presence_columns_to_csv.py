from pathlib import Path
import os
import pandas as pd
import zipfile


mri_embeddings_dir = '/gscratch/scrubbed/juampablo/corebt_dataset/MRI'
histopathology_embeddings_dir = '/gscratch/scrubbed/juampablo/corebt_dataset/Pathology'
dataset_csvs_dir = '/gscratch/kurtlab/CoreBT/experiments_mediqa/dataset_csvs'


def add_present_column():

    splits = ['train', 'val', 'test']
    for split in splits:
        dataset_csv = Path(dataset_csvs_dir) / f'{split}.csv'
        split_df = pd.read_csv(dataset_csv, dtype={"subject_id": str})
        
        split_subjects = split_df['subject_id'].tolist()

        has_histopathology_embedding_df, has_mri_embedding_df = {}, {}
        for subject_id in split_subjects:
            subject_histopathology_embedding_dir, subject_mri_embedding_dir = \
                [Path(d) / split / subject_id for d in [histopathology_embeddings_dir, mri_embeddings_dir]]

            if subject_histopathology_embedding_dir.exists():
                has_histopathology_embedding_df[subject_id] = any(subject_histopathology_embedding_dir.iterdir())
            else:
                has_histopathology_embedding_df[subject_id] = False
            
            if subject_mri_embedding_dir.exists():
                has_mri_embedding_df[subject_id] = any(subject_mri_embedding_dir.iterdir())
            else:
                has_mri_embedding_df[subject_id] = False
            
        split_df['histopathology_present'] = split_df['subject_id'].map(has_histopathology_embedding_df)
        split_df['mri_present'] = split_df['subject_id'].map(has_mri_embedding_df)

        split_df.to_csv(str(dataset_csv), index=False)


if __name__ == '__main__':
    subs=[]
    add_present_column()

    print('yep')

    