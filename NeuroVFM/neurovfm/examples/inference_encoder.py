import torch
from neurovfm.pipelines import load_encoder

# 1. Load encoder + preprocessor from your HF repo
encoder, preproc = load_encoder(
    "mlinslab/neurovfm-encoder",   # your repo ID
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# 2. Preprocess a study (directory or single file)
batch = preproc.load_study("/gscratch/kurtlab/models/NeuroVFM/neurovfm/examples/data/BraTS-GLI-00000-000-t1c.nii.gz", modality="mri")  # or "ct"

# 3. Get token-level representations
embs = encoder.embed(batch)  # shape: [N_tokens, D]

# Optionally: move to CPU / save
embs_cpu = embs.cpu()