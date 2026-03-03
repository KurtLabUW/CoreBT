import torch, os
from neurovfm.pipelines import load_encoder, load_diagnostic_head
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm

'''
module load gcc/12.3.0
python3 -m examples.embed_all_corebt


ls /gscratch/kurtlab/juampablo/corebt_mri_v2/10064774423088
10064774423088-seg.nii.gz  10064774423088-t1c.nii.gz  10064774423088-t1n.nii.gz  10064774423088-t2f.nii.gz  10064774423088-t2w.nii.gz
'''


'''
MRI_SUBJECT_DIR=/gscratch/scrubbed/juampablo/corebt/corebt_mri_v2
EMBEDDINGS_DIR=/gscratch/scrubbed/juampablo/corebt/mri_embeddings
NUM_SPLITS=1
SPLIT_NO=0
python3 -m examples.embed_all_corebt --mri_subject_dir $MRI_SUBJECT_DIR --embeddings_dir $EMBEDDINGS_DIR --num_splits $NUM_SPLITS --split_no $SPLIT_NO
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--subject_id", type=str, required=True)
    parser.add_argument("--mri_subject_dir", type=str, default="outputs/tile_embeddings")
    parser.add_argument("--embeddings_dir", type=str, default="outputs/slide_embeddings")
    parser.add_argument('--num_splits', type=int, default=1, help='Total number of jobs/splits')
    parser.add_argument('--split_no', type=int, default=0, help='Current job index (0-based)')
    args = parser.parse_args()

    os.makedirs(args.embeddings_dir, exist_ok=True)

    all_subjects = np.array(sorted([p for p in Path(args.mri_subject_dir).iterdir() if p.is_dir()]))

    indices = np.arange(len(all_subjects))
    split_indices = np.array_split(indices, args.num_splits)[args.split_no]
    subjects = all_subjects[split_indices]

    print(f'Processing split {args.split_no + 1} / {args.num_splits}, contains {len(subjects)} subjects.')

    encoder, preproc = load_encoder(
        "mlinslab/neurovfm-encoder",   
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    for ix, subdir in tqdm(enumerate(subjects), total=len(subjects)):
        subject_id= Path(subdir).name 

        save_path = os.path.join(args.embeddings_dir, f"{subject_id}.pt")

        print(f'[{ix+1}/{len(subjects)}] Obtaining slide embeddings for Subject {subject_id}')
        if os.path.exists(save_path):
            print(f'Embedding file found in {save_path}, skipping..')
            continue

        batch = preproc.load_study(subdir, modality="mri")

        # Get token-level representations
        embs = encoder.embed(batch)  # shape: [N_tokens, D]
        torch.save(embs.cpu(), save_path)

        print(f'Saved embeddings for subject {subject_id} to {save_path}')

