import os
import io
import argparse
import zipfile
import pandas as pd

import torch
import itertools
import numpy as np
from torch import nn
import json

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.utils.tensorboard as tensorboard
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, recall_score, confusion_matrix


from collections import defaultdict



def seed_torch(device, seed=7):
    # References:
    # HIPT: https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/main.py
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Make argparser
argparser = argparse.ArgumentParser(description='Linear Probe')
# Dataset
argparser.add_argument('--dataset_csv',            type=str, default='', help='The csv file indicating input samples and labels')
argparser.add_argument('--input_path',          type=str, default='', help='The input embedding files')
argparser.add_argument('--embed_dim',           type=int, default=1536, help='The dimension of the embeddings')
# Training
argparser.add_argument('--batch_size',          type=int, default=512, help='Batch size')
argparser.add_argument('--train_iters',         type=int, default=12500, help='Number of epochs')
argparser.add_argument('--lr',                  type=float, default=0.01, help='Learning rate')
argparser.add_argument('--min_lr',              type=float, default=0.0, help='Minimum learning rate')
argparser.add_argument('--optim',               type=str, default='sgd', help='Optimizer')
argparser.add_argument('--momentum',            type=float, default=0.0, help='Momentum')
argparser.add_argument('--weight_decay',        type=float, default=0.0, help='Weight decay')
argparser.add_argument('--eval_interval',       type=int, default=10000, help='Evaluation interval')
argparser.add_argument('--model_select',        type=str, default='best', help='Model selection')
argparser.add_argument('--num_workers',         type=int, default=10, help='Number of workers')
argparser.add_argument('--seed',                type=int, default=42, help='Random seed')
argparser.add_argument('--z_score',             action='store_true', default=False, help='Whether use z-score normalization')
# Output
argparser.add_argument('--output_dir',          type=str, default='outputs', help='Output directory')


def to_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    '''Convert the labels to one-hot encoding'''
    onehot = np.zeros((labels.shape[0], num_classes))
    onehot[np.arange(labels.shape[0]), labels] = 1
    return onehot


def train(model,
          train_loader,
          val_loader,
          test_loader,
          train_iters,
          lr, min_lr,
          optim,
          weight_decay,
          output_dir,
          eval_interval,
          momentum,
          **kwargs):
    """
    Train the linear probe model.

    Arguments:
    --
    model: nn.Module
        Linear probe model
    train_loader: DataLoader
        DataLoader for training set
    val_loader: DataLoader
        DataLoader for validation set
    test_loader: DataLoader
        DataLoader for test set
    train_iters: int
        Number of training iterations
    lr: float
        Learning rate
    min_lr: float
        Minimum learning rate
    optim: str
        Optimizer
    weight_decay: float
        Weight decay
    output_dir: str
        Output directory
    eval_interval: int
        Evaluation interval
    momentum: float
        Momentum
    """
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # set Tensorboard
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tensorboard_dir)

    # Set the optimizer
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer')
    print('Set the optimizer as {}'.format(optim))

    # Set the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_iters, eta_min=min_lr)

    # Set the loss function
    criterion = nn.CrossEntropyLoss()

    # Set the infinite train loader
    infinite_train_loader = itertools.cycle(train_loader)

    best_f1 = 0
    # Train the model
    print('Start training')
    for i, (embed, target) in enumerate(infinite_train_loader):

        if i >= train_iters:
            break
        

        embed, target = embed.to(device), target.to(device)

        # Forward pass
        output = model(embed)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if (i + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Iteration [{i}/{train_iters}]\tLoss: {loss.item()}\tLR: {lr}')
            writer.add_scalar('Train Loss', loss.item(), i)
            writer.add_scalar('Learning Rate', lr, i)
        # Print the loss
        if (i + 1) % eval_interval == 0 or (i + 1) == train_iters:
            print(f'Start evaluating ...')
            results = evaluate(model, criterion, val_loader, device)

            accuracy = results["global_metrics"]["accuracy"]
            f1 = results["global_metrics"]["f1_weighted"]
            precision = results["global_metrics"]["precision_macro"]
            recall = results["global_metrics"]["recall_macro"]
            auroc = results["global_metrics"]["auroc_macro"]
            auprc = results["global_metrics"]["auprc_macro"]
            print(f'Val [{i}/{train_iters}] Accuracy: {accuracy} f1: {f1} Precision: {precision} Recall: {recall} AUROC: {auroc} AUPRC: {auprc}')

            writer.add_scalar('Val Accuracy', accuracy, i)
            writer.add_scalar('Val f1', f1, i)
            writer.add_scalar('Val AUROC', auroc, i)
            writer.add_scalar('Val AUPRC', auprc, i)
            writer.add_scalar('Val Precision', precision, i)
            writer.add_scalar('Val Recall', recall, i)
            writer.add_scalar('Best f1', best_f1, i)

            if f1 > best_f1:
                print('Best f1 increase from {} to {}'.format(best_f1, f1))
                best_f1 = f1
                torch.save(model.state_dict(), f'{output_dir}/best_model.pth')

    # Save the model
    torch.save(model.state_dict(), f'{output_dir}/model.pth')

    if kwargs.get('model_select') == 'best':
        val_f1 = best_f1
        model.load_state_dict(torch.load(f'{output_dir}/best_model.pth'))
    else:
        val_f1 = f1
        model.load_state_dict(torch.load(f'{output_dir}/model.pth'))

    # Evaluate the model
    results = evaluate(model, criterion, test_loader, device)

    accuracy = results["global_metrics"]["accuracy"]
    f1 = results["global_metrics"]["f1_weighted"]
    precision = results["global_metrics"]["precision_macro"]
    recall = results["global_metrics"]["recall_macro"]
    auroc = results["global_metrics"]["auroc_macro"]
    auprc = results["global_metrics"]["auprc_macro"]
    print(f'Test Accuracy: {accuracy} f1: {f1} Precision: {precision} Recall: {recall} AUROC: {auroc} AUPRC: {auprc}')
    

    results["iteration"] = i
    results["val_f1"] = val_f1

    json_path = os.path.join(output_dir, "results.json")

    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        try:
            with open(json_path, "r") as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            print("Warning: results.json corrupted. Reinitializing.")
            all_results = []
    else:
        all_results = []

    all_results.append(results)

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)

def evaluate(model, criterion, val_loader, device):
    model.eval()

    total_loss = 0
    pred_gather, target_gather = [], []

    with torch.no_grad():
        for embed, target in val_loader:

            embed = embed.to(device)
            target = target.to(device)

            output = model(embed)
            loss = criterion(output, target)
            total_loss += loss.item()

            pred_gather.append(output.cpu().numpy())
            target_gather.append(target.cpu().numpy())

    pred_gather = np.concatenate(pred_gather)
    target_gather = np.concatenate(target_gather)
    preds = pred_gather.argmax(1)

    total_samples = len(target_gather)
    avg_loss = total_loss / len(val_loader)

    
    # Global Metrics
    accuracy = float((preds == target_gather).mean())
    f1_weighted = float(f1_score(target_gather, preds, average="weighted"))
    f1_macro = float(f1_score(target_gather, preds, average="macro"))

    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        target_gather, preds, average="macro", zero_division=0
    )

    balanced_acc = float(
        recall_score(target_gather, preds, average="macro", zero_division=0)
    )

    
    # Per-class metrics
    labels = np.unique(target_gather)

    precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
        target_gather, preds, labels=labels, zero_division=0
    )

    per_class = {}

    for i, cls in enumerate(labels):
        per_class[int(cls)] = {
            "support": int(support_c[i]),
            "fraction": float(support_c[i] / total_samples),
            "precision": float(precision_c[i]),
            "recall": float(recall_c[i]),
            "f1": float(f1_c[i]),
        }

    
    # AUROC / AUPRC
    present = np.unique(target_gather.astype(int))

    if present.size < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        y_1h = to_onehot(target_gather, pred_gather.shape[1])
        y_1h = y_1h[:, present]
        p = pred_gather[:, present]

        auroc = float(
            roc_auc_score(y_1h, p, average="macro", multi_class="ovr")
        )
        auprc = float(
            average_precision_score(y_1h, p, average="macro")
        )

    
    # Confusion Matrix
    cm = confusion_matrix(target_gather, preds, labels=labels)

    
    # Assemble structured results
    results = {
        "summary": {
            "num_samples": int(total_samples),
            "avg_loss": float(avg_loss),
        },
        "global_metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "auroc_macro": auroc,
            "auprc_macro": auprc,
        },
        "per_class_metrics": per_class,
        "confusion_matrix": {
            "labels": labels.tolist(),
            "matrix": cm.tolist(),
        },
    }

    return results



def summarize_dataset(name, dataset):
    print("\n" + "="*70)
    print(f"{name.upper()} SUMMARY")
    print("="*70)

    print(f"Total samples: {len(dataset)}")

    # Label dict
    print("Label mapping (label_dict):")
    print(dataset.label_dict)

    # Label distribution
    labels = np.array([dataset[i][2] if len(dataset[i]) == 3 else dataset[i][1] 
                       for i in range(len(dataset))])
    # Handles (mri, histo, target) or (embed, target)

    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution:")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c}")
    
    print("Unique labels in this split:", sorted(unique.tolist()))

    # If multimodal
    if hasattr(dataset, "samples"):
        print("\nFirst 5 sample IDs:")
        print(dataset.samples[:5])

    if hasattr(dataset, "embeds_mri"):
        print(f"\nMRI embeddings loaded: {len(dataset.embeds_mri)}")
    if hasattr(dataset, "embeds_histo"):
        print(f"Histo embeddings loaded: {len(dataset.embeds_histo)}")

    print("="*70 + "\n")

def main():
    args = argparser.parse_args()
    print(args)

    # set the random seed
    seed_torch(torch.device('cuda'), args.seed)
    # set the processor
    processor = ProcessorPool()
    # load the dataset
    splits = ['train', 'val', 'test']
    train_dataset, val_dataset, test_dataset = [EmbeddingDataset(args.dataset_csv, args.input_path, \
                    split=split, z_score=args.z_score, processor=processor) for split in splits]
    # set num_classes
    args.num_classes = len(train_dataset.label_dict)
    summarize_dataset("Train", train_dataset)
    summarize_dataset("Val", val_dataset)
    summarize_dataset("Test", test_dataset)
    print(f'Train: {len(train_dataset)}\tVal: {len(val_dataset)}\tTest: {len(test_dataset)}')


    print("\n================ LABEL DICTS ================")
    print("Train label_dict:", train_dataset.label_dict)
    print("Val   label_dict:", val_dataset.label_dict)
    print("Test  label_dict:", test_dataset.label_dict)
    print("=============================================\n")

    def print_split_stats(name, dataset):
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n{name} Split Label Distribution:")
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c}")
        print("Total:", len(labels))


    print_split_stats("Train", train_dataset)
    print_split_stats("Val", val_dataset)
    print_split_stats("Test", test_dataset)
    # # infinite sampler for training
    # train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset, replacement=True)

    # get training labels
    train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.log(class_counts + 1) #class_counts

    # assign weight per sample
    sample_weights = class_weights[train_labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Load the model
    model = LinearProbe(args.embed_dim, args.num_classes)

    # Train the model
    train(model, train_loader, val_loader, test_loader, **vars(args))


class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int = 768, num_classes: int = 10):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.ln(x)
        return self.fc(x)


class EmbeddingDataset(Dataset):
    def __init__(self, dataset_csv: str, zip_path: str, split: str = 'train', z_score=False, processor=None):
        df = pd.read_csv(dataset_csv, dtype={"trial_accession": str})

        #  normalize trial_accession 
        df["trial_accession"] = (
            df["trial_accession"]
            .astype(str)
            .str.strip()
            .str.replace(".0", "", regex=False)
        )

        # keep only this split
        split_df = df[df["split"] == split].copy()

        # Remove rows with NaN / empty labels
        split_df = split_df[split_df["label"].notna()]
        split_df = split_df[split_df["label"].astype(str).str.strip() != ""]
        split_df = split_df[split_df["label"].astype(str).str.lower() != "nan"]

        # If mri_present exists, filter by it (MRI-only run)
        if "mri_present" in split_df.columns:
            split_df = split_df[split_df["mri_present"] == True].copy()

        #  load embeddings FIRST (source of truth) 
        self.processor = processor
        self.embeds = processor.load_embeddings_from_zip(zip_path, split)
        embed_keys = set(self.embeds.keys())

        #  filter rows to those actually in the zip 
        before = len(split_df)
        split_df = split_df[split_df["trial_accession"].isin(embed_keys)].copy()
        after = len(split_df)

        if after < before:
            missing = before - after
            # show a few missing IDs for debugging
            missing_ids = (
                df[df["split"] == split]["trial_accession"]
                .astype(str).str.strip().str.replace(".0", "", regex=False)
            )
            # compute missing within split+mri_present+label filtered set
            tmp = df[df["split"] == split].copy()
            tmp["trial_accession"] = tmp["trial_accession"].astype(str).str.strip().str.replace(".0", "", regex=False)
            tmp = tmp[tmp["label"].notna()]
            if "mri_present" in tmp.columns:
                tmp = tmp[tmp["mri_present"] == True]
            miss_list = [x for x in tmp["trial_accession"].tolist() if x not in embed_keys][:10]

            print(f"[Dataset Init:{split}] Dropped {missing} rows because trial_accession not found in zip embeddings.")
            print(f"[Dataset Init:{split}] Example missing trial_accession: {miss_list}")

        #  finalize samples/labels 
        self.samples = split_df["trial_accession"].astype(str).str.strip().tolist()
        self.labels = split_df["label"].astype(int).tolist()

        # If labels are already encoded (0..C-1), keep identity mapping
        # uniq = sorted(set(self.labels))
        # self.label_dict = {lbl: i for i, lbl in enumerate(uniq)}
        all_labels = sorted(pd.read_csv(dataset_csv)['label'].dropna().unique())
        self.label_dict = {label: i for i, label in enumerate(all_labels)}

        self.z_score = z_score

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.labels[index]

        # (optional) ultra-safe normalize again
        sample = str(sample).strip().replace(".0", "")

        embed = self.embeds[sample].float()

        if self.z_score:
            embed = (embed - embed.mean()) / (embed.std() + 1e-8)

        target = self.label_dict[target]
        return embed, target


class ProcessorPool:

    def __init__(self, pooling: str = 'mean'):
        """
        pooling:
            None   -> keep original tensor
            "mean" -> mean pool over tokens
            "max"  -> max pool over tokens
            "cls"  -> take first token
        """
        self.pooling = pooling

    def get_sample_name(self, path):
        return os.path.basename(path).replace('.pt', '')

    def pool_tensor(self, tensor: torch.Tensor):
        """
        Handles:
            [N_tokens, D]
            [D] (already pooled)
        """

        # Already pooled
        if tensor.dim() == 1:
            return tensor

        if tensor.dim() != 2:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

        if self.pooling is None:
            return tensor

        if self.pooling == "mean":
            return tensor.mean(dim=0)

        if self.pooling == "max":
            return tensor.max(dim=0).values

        if self.pooling == "cls":
            return tensor[0]

        raise ValueError(f"Unknown pooling method: {self.pooling}")

    def load_embeddings_from_zip(self, zip_path, split=None):

        loaded_tensors = {}

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:

            file_list = zip_ref.infolist()
            print(f"Found {len(file_list)} files in zip")

            for file_info in tqdm(file_list):
                if not file_info.filename.endswith('.pt'):
                    continue

                file_bytes = zip_ref.read(file_info.filename)
                byte_stream = io.BytesIO(file_bytes)

                tensor = torch.load(byte_stream, map_location="cpu")

                tensor = self.pool_tensor(tensor)

                sample_name = self.get_sample_name(file_info.filename)

                loaded_tensors[sample_name] = tensor

        print(f"Loaded {len(loaded_tensors)} embeddings")
        print(f"Embeddings shape: {tensor.size()}") ###########
        return loaded_tensors


class Processor:

    def get_sample_name(self, path):
        return os.path.basename(path).replace('.pt', '')
    
    def load_embeddings_from_zip(self, zip_path, split):
        loaded_tensors = {}
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(len(zip_ref.infolist()))

            for file_info in tqdm(zip_ref.infolist()):
                
                if file_info.filename.endswith('.pt'):# and split in file_info.filename:
                    file_bytes = zip_ref.read(file_info.filename)
                    byte_stream = io.BytesIO(file_bytes)
                    tensor = torch.load(byte_stream)
                    sample_name = self.get_sample_name(file_info.filename)
                    loaded_tensors[sample_name] = tensor
        # print(list(loaded_tensors.keys()))
        return loaded_tensors


if __name__ == '__main__':
    main()
    # processor = Processor()
    # dataset_csv='/gscratch/kurtlab/CoreBT/dataset_utils/fusion/corebt_fusion_dataset_WHOGrade_case_sharedtest.csv'
    # zip_path= '/gscratch/scrubbed/juampablo/corebt/mri_embeddings.zip'
    # EmbeddingDataset(dataset_csv, zip_path, \
    #                 split='train', z_score=False, processor=processor)
# EMBED_DIM=1536


# DATASET_CSV=/gscratch/kurtlab/CoreBT/dataset_utils/linear_probe_mri/corebt_mri_multiclass_WHOGrade_case.csv
# ZIP_PATH=/gscratch/scrubbed/juampablo/corebt/mri_embeddings.zip
# OUTPUT_DIR=/gscratch/kurtlab/CoreBT/NeuroVFM/neurovfm/experiments/corebt_linear_probe
# EMBED_DIM=768
# BATCH_SIZE=32
# TRAIN_ITERS=1200
# LR=0.001
# MIN_LR=0.0
# OPTIM=adam
# MOMENTUM=0.0
# WEIGHT_DECAY=1e-4
# EVAL_INTERVAL=200
# NUM_WORKERS=4
# SEED=42

# python3 -m linear_probe.corebt_mri_main --dataset_csv $DATASET_CSV --input_path $ZIP_PATH --embed_dim $EMBED_DIM --batch_size $BATCH_SIZE --train_iters $TRAIN_ITERS --lr $LR --min_lr $MIN_LR --optim $OPTIM --momentum $MOMENTUM --weight_decay $WEIGHT_DECAY --eval_interval $EVAL_INTERVAL --num_workers $NUM_WORKERS --seed $SEED --output_dir $OUTPUT_DIR