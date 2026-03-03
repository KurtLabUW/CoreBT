import os, argparse, glob
import h5py
import numpy as np
import huggingface_hub
from gigapath.pipeline import load_tile_slide_encoder
from gigapath.pipeline import run_inference_with_tile_encoder
import timm

assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

def save_prov_gigapath_embeddings(
    tile_encoder_outputs,
    slide_path,
    output_dir,
    compression_level=4,
):
    """
    Save Prov-GigaPath tile embeddings + coordinates to HDF5.
     Inspired by https://github.com/prov-gigapath/prov-gigapath/issues/73

    Parameters
    ----------
    tile_encoder_outputs : dict
        Output of run_inference_with_tile_encoder
        Must contain:
            - "tile_embeds" : torch.Tensor [N, D]
            - "coords"      : torch.Tensor [N, 2]

    slide_path : str
        Path to original slide (used to name output file)

    output_dir : str
        Directory where .h5 file will be saved

    compression_level : int
        Gzip compression level (0-9)
    """

    os.makedirs(output_dir, exist_ok=True)

    slide_id = os.path.splitext(os.path.basename(slide_path))[0]
    save_path = os.path.join(output_dir, f"{slide_id}.h5")


    # Convert to numpy
    features = tile_encoder_outputs["tile_embeds"].cpu().numpy()
    coords = tile_encoder_outputs["coords"].cpu().numpy()

    print(f"Saving slide: {slide_id}")
    print(f"Features shape: {features.shape}")
    print(f"Coords shape:   {coords.shape}")


    # Write HDF5
    with h5py.File(save_path, "w") as f:

        # Tile embeddings
        f.create_dataset(
            "features",
            data=features,
            chunks=(256, features.shape[1]),
            compression="gzip",
            compression_opts=compression_level,
        )

        # Tile coordinates
        f.create_dataset(
            "coords",
            data=coords,
            chunks=(256, 2),
            compression="gzip",
            compression_opts=compression_level,
        )

        # Metadata
        f.attrs["slide_id"] = slide_id
        f.attrs["encoder"] = "prov-gigapath"
        f.attrs["feature_dim"] = features.shape[1]
        f.attrs["num_tiles"] = features.shape[0]

    print(f"Saved → {save_path}")

    return save_path



def extract_and_save_tile_embeddings(args):
    image_paths = glob.glob(os.path.join(args.slide_dir, "*.png"))
    total_images = len(image_paths)
    print(f"Found {total_images} image tiles")

    if args.stride > 1:
            # This takes every Nth element: [start:stop:step]
            image_paths = image_paths[::args.stride]
            print(f"Downsampled to {len(image_paths)} tiles (Stride: {args.stride})")

    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, checkpoint_path='/gscratch/kurtlab/jehr/torch_cache/huggingface/hub/models--prov-gigapath--prov-gigapath/snapshots/eba85dd46097c3eedfcc2a3a9a930baecb6bcc19/pytorch_model.bin')
    
    print(f'Loaded encoders!')
    tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)

    for k in tile_encoder_outputs.keys():
        print(f"tile_encoder_outputs[{k}].shape: {tile_encoder_outputs[k].shape}")

    save_prov_gigapath_embeddings(
        tile_encoder_outputs=tile_encoder_outputs,
        slide_path=args.slide_dir,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", type=str, required=True)
    parser.add_argument("--stride", type=int, default=1,  help="Step size for downsampling (e.g., 2 picks every second image)")
    parser.add_argument("--output_dir", type=str, default="outputs/embeddings")
    args = parser.parse_args()

    args.slide_dir = f"/gscratch/scrubbed/juampablo/corebt/corebt_pathology_tiles/{args.subject_id}_lv0/output/{args.subject_id}.svs"

    extract_and_save_tile_embeddings(args)

    # SUBJECT_ID=C013_A1.5_HE
    # python3 -m tile_filtering.embed_and_save_tiles --subject_id $SUBJECT_ID
    # Running inference with tile encoder: 100%|███████████████████████████| 38/38 [13:35<00:00, 21.46s/it]
    # tile_encoder_outputs[tile_embeds].shape: torch.Size([4741, 1536])
    # tile_encoder_outputs[coords].shape: torch.Size([4741, 2])
    # Saving slide: C013_A1.5_HE
    # Features shape: (4741, 1536)
    # Coords shape:   (4741, 2)
    # Saved → outputs/embeddings/C013_A1.5_HE.h5