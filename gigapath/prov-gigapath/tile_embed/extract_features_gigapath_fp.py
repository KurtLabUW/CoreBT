import os
import time
import argparse
import h5py
import torch
import numpy as np
import timm
import openslide
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from tile_embed.file_utils import save_hdf5
from tile_embed.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP

# Assuming these are available in your local environment from the second script
# from models import get_encoder 
# from .file_utils import save_hdf5
# from .dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_gigapath_encoder():
    """Initializes the Prov-GigaPath model and its specific transforms."""
    print("Loading Prov-GigaPath from HuggingFace...")
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, checkpoint_path='/gscratch/kurtlab/jehr/torch_cache/huggingface/hub/models--prov-gigapath--prov-gigapath/snapshots/eba85dd46097c3eedfcc2a3a9a930baecb6bcc19/pytorch_model.bin')
    model = model.to(device)
    model.eval()

    # Standard GigaPath transforms
    img_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, img_transforms

def compute_w_loader(output_path, loader, model, verbose=0):
    if verbose > 0:
        print(f'Processing a total of {len(loader)} batches')

    with h5py.File(output_path, "w") as f:
        # We will create datasets on the first batch to handle dynamic sizes
        dset_features = None
        dset_coords = None
        
        for count, data in enumerate(tqdm(loader)):
            with torch.inference_mode():    
                batch = data['img'].to(device, non_blocking=True)
                coords = data['coord'].numpy().astype(np.int32)
                
                # Forward Pass
                features = model(batch) 
                features = features.cpu().numpy().astype(np.float32)

                if dset_features is None:
                    # Initialize HDF5 datasets based on first batch shape
                    dset_features = f.create_dataset("features", data=features, 
                                                     maxshape=(None, features.shape[1]), 
                                                     chunks=(256, features.shape[1]), 
                                                     compression="gzip")
                    dset_coords = f.create_dataset("coords", data=coords, 
                                                   maxshape=(None, 2), 
                                                   chunks=(256, 2), 
                                                   compression="gzip")
                else:
                    # Append to HDF5
                    dset_features.resize(dset_features.shape[0] + features.shape[0], axis=0)
                    dset_features[-features.shape[0]:] = features
                    
                    dset_coords.resize(dset_coords.shape[0] + coords.shape[0], axis=0)
                    dset_coords[-coords.shape[0]:] = coords
    
    return output_path

'''
DATA_H5_DIR=/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/corebt_tiles/patches
DATA_SLIDE_DIR=/gscratch/scrubbed/juampablo/corebt/corebt_pathology
CSV_PATH=/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/corebt_tiles/process_list_autogen.csv
FEAT_DIR=/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/tile_embeddings
NUM_SPLITS=20
SPLIT_NO=0
python3 -m tile_embed.extract_features_gigapath_fp --data_h5_dir $DATA_H5_DIR --data_slide_dir $DATA_SLIDE_DIR --csv_path $CSV_PATH --feat_dir $FEAT_DIR --split_no $SPLIT_NO --num_splits $NUM_SPLITS
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GigaPath Feature Extraction')
    parser.add_argument('--data_h5_dir', type=str, required=True, help='Dir containing patch coords .h5')
    parser.add_argument('--data_slide_dir', type=str, required=True, help='Dir containing .svs files')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV with slide IDs')
    parser.add_argument('--feat_dir', type=str, default="outputs/gigapath_features")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--slide_ext', type=str, default='.svs')

    parser.add_argument('--num_splits', type=int, default=1, help='Total number of jobs/splits')
    parser.add_argument('--split_no', type=int, default=0, help='Current job index (0-based)')

    args = parser.parse_args()

    # 1. Initialize GigaPath
    model, img_transforms = get_gigapath_encoder()

    # 2. Setup Dataset
    bags_dataset = Dataset_All_Bags(args.csv_path, num_splits=args.num_splits, split_no=args.split_no)
    
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)

    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    for bag_candidate_idx in range(len(bags_dataset)):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        print(f"\n[{bag_candidate_idx+1}/{len(bags_dataset)}] Processing: {slide_id}")

        h5_file_path = os.path.join(args.data_h5_dir, f"{slide_id}.h5")
        slide_file_path = os.path.join(args.data_slide_dir, f"{slide_id}{args.slide_ext}")
        output_h5_path = os.path.join(args.feat_dir, 'h5_files', f"{slide_id}.h5")
        output_pt_path = os.path.join(args.feat_dir, 'pt_files', f"{slide_id}.pt")

        if os.path.exists(output_pt_path):
            print(f"Skipping {slide_id}, already exists.")
            continue

        # 3. Load WSI and create DataLoader
        wsi = openslide.open_slide(slide_file_path)
        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, img_transforms=img_transforms)
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)

        # 4. Extract and Save
        compute_w_loader(output_h5_path, loader=loader, model=model, verbose=1)

        # 5. Save .pt file for downstream MIL training
        with h5py.File(output_h5_path, "r") as f:
            features = torch.from_numpy(f['features'][:])
            torch.save(features, output_pt_path)
            print(f"Saved features shape: {features.shape}")


