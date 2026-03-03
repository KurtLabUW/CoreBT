import argparse
import gigapath.slide_encoder as slide_encoder
from gigapath.pipeline import run_inference_with_slide_encoder
import h5py
import torch
import os

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



# def extract_and_save_slide_embeddings(args):
#     slide_encoder_model = slide_encoder.create_model('/gscratch/kurtlab/jehr/torch_cache/huggingface/hub/models--prov-gigapath--prov-gigapath/snapshots/eba85dd46097c3eedfcc2a3a9a930baecb6bcc19/slide_encoder.pth',
#                                                          "gigapath_slide_enc12l768d", 1536, global_pool=True)

#     tile_encoder_outputs = load_prov_gigapath_h5(args.h5_path, device="cuda")

#     slide_embeds = run_inference_with_slide_encoder(
#         slide_encoder_model=slide_encoder_model,
#         **tile_encoder_outputs
#     )

#     print(slide_embeds.keys())

#     for k, v in slide_embeds.items():
#         if hasattr(v, "shape"):
#             print(f"{k:<20} shape={tuple(v.shape)}  dtype={v.dtype}  device={v.device}")



def extract_and_save_slide_embeddings(args):

    os.makedirs(args.save_dir, exist_ok=True)

    slide_encoder_model = slide_encoder.create_model('/gscratch/kurtlab/jehr/torch_cache/huggingface/hub/models--prov-gigapath--prov-gigapath/snapshots/eba85dd46097c3eedfcc2a3a9a930baecb6bcc19/slide_encoder.pth',
                                                         "gigapath_slide_enc12l768d", 1536, global_pool=True)


    tile_encoder_outputs = load_prov_gigapath_h5(args.h5_path, device="cuda")

    with torch.no_grad():
        slide_embeds = run_inference_with_slide_encoder(
            slide_encoder_model=slide_encoder_model,
            **tile_encoder_outputs
        )

    print(slide_embeds.keys())

    save_dict = {}

    for k, v in slide_embeds.items():
        if hasattr(v, "shape"):
            print(f"{k:<20} shape={tuple(v.shape)}  dtype={v.dtype}  device={v.device}")

            # remove batch dim if present
            if v.dim() == 2 and v.size(0) == 1:
                v = v.squeeze(0)

            save_dict[k] = v.cpu()

    save_path = os.path.join(args.save_dir, f"{args.subject_id}.pt")
    torch.save(save_dict, save_path)

    print(f"\nSaved all layer embeddings → {save_path}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", type=str, required=True)
    parser.add_argument("--tile_embeddings_dir", type=str, default="outputs/tile_embeddings")
    parser.add_argument("--save_dir", type=str, default="outputs/slide_embeddings")
    args = parser.parse_args()

    args.h5_path = os.path.join(args.tile_embeddings_dir, f'{args.subject_id}.h5')

    extract_and_save_slide_embeddings(args)
    
    # python3 -m tile_filtering.slide_embed --subject_id $SUBJECT_ID 


    # dict_keys(['layer_0_embed', 'layer_1_embed', 'layer_2_embed', 'layer_3_embed', 'layer_4_embed', 'layer_5_embed', 'layer_6_embed', 'layer_7_embed', 'layer_8_embed', 'layer_9_embed', 'layer_10_embed', 'layer_11_embed', 'layer_12_embed', 'last_layer_embed'])
    # layer_0_embed        shape=(1, 768)  dtype=torch.float32  device=cpu
    # layer_1_embed        shape=(1, 768)  dtype=torch.float32  device=cpu
    # layer_2_embed        shape=(1, 768)  dtype=torch.float32  device=cpu
    # layer_3_embed        shape=(1, 768)  dtype=torch.float32  device=cpu
    # layer_4_embed        shape=(1, 768)  dtype=torch.float32  device=cpu
    # layer_5_embed        shape=(1, 768)  dtype=torch.float32  device=cpu
    # layer_6_embed        shape=(1, 768)  dtype=torch.float32  device=cpu
    # layer_7_embed        shape=(1, 768)  dtype=torch.float32  device=cpu
    # layer_8_embed        shape=(1, 768)  dtype=torch.float32  device=cpu
    # layer_9_embed        shape=(1, 768)  dtype=torch.float32  device=cpu
    # layer_10_embed       shape=(1, 768)  dtype=torch.float32  device=cpu
    # layer_11_embed       shape=(1, 768)  dtype=torch.float32  device=cpu
    # layer_12_embed       shape=(1, 768)  dtype=torch.float32  device=cpu
    # last_layer_embed     shape=(1, 768)  dtype=torch.float32  device=cpu
