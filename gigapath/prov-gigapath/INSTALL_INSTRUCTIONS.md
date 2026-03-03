## GIGAPATH ENVIRONMENT RECREATION INSTRUCTIONS 
<!--  -->
### 1. CREATE ENVIRONMENT
```bash
conda create -n gigapath python=3.9 -y conda activate gigapath
```

### 2. Install base pytorch and numpy 

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118) pip install \"numpy<2.0.0\"
```

### 3. Compile FLASH-ATTENTION on GPU Node (always painful tbh)
MAX_JOBS=4 prevents the \"Killed\" OOM error on Klone. TORCH_CUDA_ARCH_LIST targets A100 (8.0), RTX3090 (8.6), and L40 (8.9).
```bash
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=\"8.0;8.6;8.9\" pip install flash-attn==2.5.8 --no-build-isolation
```

### 4. INSTALL PATHOLOGY SLIDE TOOLS
```bash
pip install openslide-python openslide-bin opencv-python scikit-image monai
```

### 5. INSTALL FULL MODEL STACK
```bash
pip install xformers==0.0.28.post3 --no-depspip install timm==0.9.12 transformers==4.36.2 einops==0.8.2 \\wandb lifelines scikit-survival scikit-learn \\omegaconf torchmetrics==0.10.3 fvcore iopath webdataset \\huggingface-hub ninja==1.11.1.1 h5py pandas pillow tqdm \\matplotlib tensorboard fairscale packaging==23.2
```

### 6. VERIFY INSTALLATION
```bash
python -c \"import torch; import flash_attn; import openslide; import numpy; print(f'NumPy version: {numpy.__version__}'); print('--- GIGAPATH STATUS: READY ---')\"
```


