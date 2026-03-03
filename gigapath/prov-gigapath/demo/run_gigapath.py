# %% [markdown]
# ## Prov-GigaPath Demo
# 
# This notebook provides a quick walkthrough of the Prov-GigaPath models. We will start by demonstrating how to download the Prov-GigaPath models from HuggingFace. Next, we will show an example of pre-processing a slide. Finally, we will demonstrate how to run Prov-GigaPath on the sample slide.
# 
# ### Prepare HF Token
# 
# To begin, please request access to the model from our HuggingFace repository: https://huggingface.co/prov-gigapath/prov-gigapath.
# 
# Once approved, set the HF_TOKEN to access the model.

import os
import h5py
import numpy as np


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




# %%

if __name__ == '__main__':

    import os

    # Please set your Hugging Face API token
    os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"

    assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

    # %% [markdown]
    # ### Download a sample slide

    # %%
    import huggingface_hub

    local_dir = os.path.join(os.path.expanduser("~"), ".cache/")
    # huggingface_hub.hf_hub_download("prov-gigapath/prov-gigapath", filename="sample_data/PROV-000-000001.ndpi", local_dir=local_dir, force_download=True)
    slide_path = os.path.join(local_dir, "sample_data/PROV-000-000001.ndpi")
    slide_path='/pscratch/sd/j/jehr/MSFT/CoreBT/corebt-baseline/prov-gigapath/sample_data/PROV-000-000001.ndpi'

    # %% [markdown]
    # ### Tiling
    # 
    # Whole-slide images are giga-pixel in size. To efficiently process these enormous images, we use a tiling technique that divides them into smaller, more manageable tile images. As an example, we demonstrate how to process a single slide.
    # 
    # NOTE: Prov-GigaPath is trained with slides preprocessed at 0.5 MPP. Ensure that you use the appropriate level for the 0.5 MPP.

    # %%
    from gigapath.pipeline import tile_one_slide

    tmp_dir = 'outputs/preprocessing/'
    tile_one_slide(slide_path, save_dir=tmp_dir, level=1)

    # %% [markdown]
    # ### Load the tile images

    # %%
    import os

    # load image tiles
    slide_dir = "outputs/preprocessing/output/" + os.path.basename(slide_path) + "/"
    image_paths = [os.path.join(slide_dir, img) for img in os.listdir(slide_dir) if img.endswith('.png')]

    print(f"Found {len(image_paths)} image tiles")

    # %% [markdown]
    # ### Load the Prov-GigaPath model (tile and slide encoder models)

    # %%
    from gigapath.pipeline import load_tile_slide_encoder

    # Load the tile and slide encoder models
    # NOTE: The CLS token is not trained during the slide-level pretraining.
    # Here, we enable the use of global pooling for the output embeddings.
    tile_encoder, slide_encoder_model = load_tile_slide_encoder(global_pool=True,
                                                                local_tile_encoder_path='/global/homes/j/jehr/.cache/huggingface/hub/models--prov-gigapath--prov-gigapath/snapshots/eba85dd46097c3eedfcc2a3a9a930baecb6bcc19/pytorch_model.bin',
                                                                local_slide_encoder_path="/global/homes/j/jehr/.cache/huggingface/hub/models--prov-gigapath--prov-gigapath/snapshots/eba85dd46097c3eedfcc2a3a9a930baecb6bcc19/slide_encoder.pth")

    # %% [markdown]
    # ### Run tile-level inference

    # %%
    from gigapath.pipeline import run_inference_with_tile_encoder

    tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)

    for k in tile_encoder_outputs.keys():
        print(f"tile_encoder_outputs[{k}].shape: {tile_encoder_outputs[k].shape}")

    save_prov_gigapath_embeddings(
        tile_encoder_outputs=tile_encoder_outputs,
        slide_path=slide_path,
        output_dir="outputs/embeddings"
    )
    # raise ValueError(f'stop here')

    # %% [markdown]
    # ### Run slide-level inference

    # %%
    from gigapath.pipeline import run_inference_with_slide_encoder
    # run inference with the slide encoder
    slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder_model, **tile_encoder_outputs)
    print(slide_embeds.keys()) # dict_keys(['layer_0_embed', 'layer_1_embed', 'layer_2_embed', 'layer_3_embed', 'layer_4_embed', 'layer_5_embed', 'layer_6_embed', 'layer_7_embed', 'layer_8_embed', 'layer_9_embed', 'layer_10_embed', 'layer_11_embed', 'layer_12_embed', 'last_layer_embed'])

    for k, v in slide_embeds.items():
        if hasattr(v, "shape"):
            print(f"{k:<20} shape={tuple(v.shape)}  dtype={v.dtype}  device={v.device}")
        else:
            print(f"{k:<20} type={type(v)}")
