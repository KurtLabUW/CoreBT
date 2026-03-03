import argparse
import gigapath.slide_encoder as slide_encoder
from gigapath.pipeline import run_inference_with_slide_encoder
import h5py
import torch
import os
from tqdm import tqdm
import glob
import numpy as np
from pathlib import Path



def load_prov_gigapath_h5(h5_path, device="cpu"):
    """
    Load tile embeddings + coordinates from saved HDF5 file
    and return a dict compatible with run_inference_with_slide_encoder.
    """
    with h5py.File(h5_path, "r") as f:
        features = torch.tensor(f["features"][:], dtype=torch.float32)
        coords = torch.tensor(f["coords"][:], dtype=torch.float32)

    return {
        "tile_embeds": features.to(device),
        "coords": coords.to(device)
    }


def extract_and_save_slide_embeddings(slide_encoder_model, tile_encoder_outputs, save_path):
    with torch.no_grad():
        slide_embeds = run_inference_with_slide_encoder(
            slide_encoder_model=slide_encoder_model,
            **tile_encoder_outputs
        )

    # print(slide_embeds.keys())

    save_dict = {}
    for k, v in slide_embeds.items():
        if hasattr(v, "shape"):
            print(f"{k:<20} shape={tuple(v.shape)}  dtype={v.dtype}  device={v.device}")

            # remove batch dim if present
            if v.dim() == 2 and v.size(0) == 1:
                v = v.squeeze(0)

            save_dict[k] = v.cpu()

    torch.save(save_dict, save_path)

    print(f"\nSaved slide embeddings to {save_path}")


'''
module load gcc/11
TILE_EMBEDDINGS_DIR=/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/tile_embeddings/h5_files
SLIDE_EMBEDDINGS_DIR=/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/slide_embeddings
NUM_SPLITS=1
SPLIT_NO=0

python3 -m slide_embed.embed_all_slides --tile_embeddings_dir $TILE_EMBEDDINGS_DIR --slide_embeddings_dir $SLIDE_EMBEDDINGS_DIR --num_splits $NUM_SPLITS --split_no $SPLIT_NO
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--subject_id", type=str, required=True)
    parser.add_argument("--tile_embeddings_dir", type=str, default="outputs/tile_embeddings")
    parser.add_argument("--slide_embeddings_dir", type=str, default="outputs/slide_embeddings")
    parser.add_argument('--num_splits', type=int, default=1, help='Total number of jobs/splits')
    parser.add_argument('--split_no', type=int, default=0, help='Current job index (0-based)')
    args = parser.parse_args()

    os.makedirs(args.slide_embeddings_dir, exist_ok=True)
    # args.h5_path = os.path.join(args.tile_embeddings_dir, f'{args.subject_id}.h5')

    all_h5= np.array(sorted(glob.glob(os.path.join(args.tile_embeddings_dir, '*.h5'))))

    indices = np.arange(len(all_h5))
    split_indices = np.array_split(indices, args.num_splits)[args.split_no]
    h5_paths = all_h5[split_indices]

    print(f'Processing split {args.split_no + 1} / {args.num_splits}, contains {len(h5_paths)} subjects.')

    slide_encoder_model = slide_encoder.create_model('/gscratch/kurtlab/jehr/torch_cache/huggingface/hub/models--prov-gigapath--prov-gigapath/snapshots/eba85dd46097c3eedfcc2a3a9a930baecb6bcc19/slide_encoder.pth',
                                                         "gigapath_slide_enc12l768d", 1536, global_pool=True)

    for ix, h5_path in tqdm(enumerate(h5_paths), total=len(h5_paths)):
        subject_id= Path(h5_path).stem 
        save_path = os.path.join(args.slide_embeddings_dir, f"{subject_id}.pt")

        print(f'[{ix+1}/{len(h5_paths)}] Obtaining slide embeddings for Subject {subject_id}')
        if os.path.exists(save_path):
            print(f'Embedding file found in {save_path}, skipping..')
            continue
        tile_encoder_outputs = load_prov_gigapath_h5(h5_path, device="cuda")

        extract_and_save_slide_embeddings(slide_encoder_model=slide_encoder_model, 
                                          tile_encoder_outputs=tile_encoder_outputs,
                                          save_path=save_path
                                          )
    