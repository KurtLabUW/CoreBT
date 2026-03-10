import os
import io
import argparse
import zipfile
import pandas as pd

import torch
import itertools
import numpy as np
from torch import nn

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.utils.tensorboard as tensorboard
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, recall_score, confusion_matrix

import json
from collections import defaultdict


from models import RobustGatedFusion
import plotext as plt


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
argparser.add_argument('--zip_path_mri',          type=str, default='', help='The input MRI embedding files')
argparser.add_argument('--zip_path_histo',          type=str, default='', help='The input Histopathology embedding files')
# argparser.add_argument('--embed_dim',           type=int, default=1536, help='The dimension of the embeddings')
argparser.add_argument('--histo_embed_dim',           type=int, default=768, help='The dimension of the embeddings')
argparser.add_argument('--mri_embed_dim',           type=int, default=768, help='The dimension of the embeddings')
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

argparser.add_argument('--mri_probe_path',          type=str,  help='Path to pretrained linear probe for MRI')
argparser.add_argument('--histo_probe_path',          type=str,  help='Path to pretrained linear probe for pathology')


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    # Tensorboard
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = torch.utils.tensorboard.SummaryWriter(tensorboard_dir)

    # Optimizer
    if optim == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError('Invalid optimizer')

    print(f'Set optimizer: {optim}')

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_iters,
        eta_min=min_lr
    )

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


    infinite_train_loader = itertools.cycle(train_loader)

    best_f1 = 0
    last_f1 = 0

    print('Start training')
    train_losses = []
    loss_steps = []

    val_f1s = []
    val_steps = []
    for i, (embed_mri, embed_histo, modality_mask, target) in enumerate(infinite_train_loader):

        if i >= train_iters:
            break

        embed_mri = embed_mri.to(device)
        embed_histo = embed_histo.to(device)
        modality_mask = modality_mask.to(device)
        target = target.to(device)


        output = model(
            embed_mri=embed_mri,
            embed_histo=embed_histo,
            mask=modality_mask
        )


        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if (i + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Iter [{i+1}/{train_iters}] '
                  f'Loss: {loss.item():.4f} '
                  f'LR: {current_lr:.6f}')
            train_losses.append(loss.item())
            loss_steps.append(i + 1)

            writer.add_scalar('Train/Loss', loss.item(), i)
            writer.add_scalar('Train/LR', current_lr, i)

        
        # Evaluation
        if (i + 1) % eval_interval == 0 or (i + 1) == train_iters:

            print('Evaluating on validation set...')
            results = evaluate(model, criterion, val_loader, device)

            accuracy = results["global_metrics"]["accuracy"]
            f1 = results["global_metrics"]["f1_weighted"]
            precision = results["global_metrics"]["precision_macro"]
            recall = results["global_metrics"]["recall_macro"]
            auroc = results["global_metrics"]["auroc_macro"]
            auprc = results["global_metrics"]["auprc_macro"]

            last_f1 = f1
            val_f1s.append(f1)
            val_steps.append(i + 1)

            print(f'Val [{i+1}/{train_iters}] '
                  f'Acc: {accuracy:.4f} '
                  f'F1: {f1:.4f} '
                  f'Prec: {precision:.4f} '
                  f'Rec: {recall:.4f} '
                  f'AUROC: {auroc:.4f} '
                  f'AUPRC: {auprc:.4f}')

            writer.add_scalar('Val/Accuracy', accuracy, i)
            writer.add_scalar('Val/F1_weighted', f1, i)
            writer.add_scalar('Val/AUROC', auroc, i)
            writer.add_scalar('Val/AUPRC', auprc, i)
            writer.add_scalar('Val/Precision', precision, i)
            writer.add_scalar('Val/Recall', recall, i)
            writer.add_scalar('Val/Best_F1', best_f1, i)

            if f1 > best_f1:
                print(f'Best F1 improved: {best_f1:.4f} → {f1:.4f}')
                best_f1 = f1
                torch.save(model.state_dict(),
                           os.path.join(output_dir, 'best_model.pth'))

    # Save final model
    torch.save(model.state_dict(),
               os.path.join(output_dir, 'model.pth'))

    plt.clf()                     # clear previous plots
    plt.plotsize(100, 25)         # width, height in terminal chars
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.plot(val_steps, val_f1s)

    plt.show()
    
    # Load Best or Final
    if kwargs.get('model_select') == 'best':
        model.load_state_dict(
            torch.load(os.path.join(output_dir, 'best_model.pth'))
        )
        val_f1 = best_f1
    else:
        model.load_state_dict(
            torch.load(os.path.join(output_dir, 'model.pth'))
        )
        val_f1 = last_f1

    # print(f'Alpha: {model.alpha.item():.4f}')
    # print(f'Beta: {model.beta.item():.4f}')

    print('Evaluating on test set (all ablations)...')

    # Run All Ablations
    eval_configs = {
        "fusion_full": dict(ablate_mri=False, ablate_histo=False),
        "mri_only": dict(ablate_mri=False, ablate_histo=True),
        "histo_only": dict(ablate_mri=True, ablate_histo=False),
        "both_ablated": dict(ablate_mri=True, ablate_histo=True),
    }

    ablation_results = {}

    for name, cfg in eval_configs.items():

        print(f"\n=== {name.upper()} ===")

        res = evaluate_ablation(
            model,
            criterion,
            test_loader,
            device,
            ablate_mri=cfg["ablate_mri"],
            ablate_histo=cfg["ablate_histo"],
        )

        gm = res["global_metrics"]

        print(
            f'Accuracy: {gm["accuracy"]:.4f}  '
            f'F1(w): {gm["f1_weighted"]:.4f}  '
            f'Precision: {gm["precision_macro"]:.4f}  '
            f'Recall: {gm["recall_macro"]:.4f}  '
            f'AUROC: {gm["auroc_macro"]:.4f}  '
            f'AUPRC: {gm["auprc_macro"]:.4f}'
        )

        ablation_results[name] = res

    final_results = {
        "iteration": i,
        "val_f1": val_f1,
        "test_results": ablation_results,
        "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

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
        all_results.append(final_results)

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nAll ablation results saved.")

    print("Training complete.")



def evaluate_ablation(
    model,
    criterion,
    loader,
    device,
    ablate_mri=False,
    ablate_histo=False
):

    model.eval()

    total_loss = 0
    pred_gather = []
    target_gather = []

    with torch.no_grad():

        for embed_mri, embed_histo, modality_mask, target in loader:

            embed_mri = embed_mri.to(device)
            embed_histo = embed_histo.to(device)
            modality_mask = modality_mask.to(device)
            target = target.to(device)

            
            # Force ablation via mask override
            if ablate_mri:
                modality_mask[:, 0] = 0.0

            if ablate_histo:
                modality_mask[:, 1] = 0.0

            # Forward pass
            output = model(
                embed_mri=embed_mri,
                embed_histo=embed_histo,
                mask=modality_mask
            )

            loss = criterion(output, target)
            total_loss += loss.item()
            
            pred_gather.append(output.detach().cpu().numpy())
            target_gather.append(target.detach().cpu().numpy())

    
    # Concatenate
    pred_gather = np.concatenate(pred_gather)
    target_gather = np.concatenate(target_gather)

    preds = pred_gather.argmax(1)

    total_samples = len(target_gather)
    avg_loss = total_loss / len(loader)

    
    # Global Metrics
    accuracy = float((preds == target_gather).mean())

    f1_weighted = float(
        f1_score(target_gather, preds, average="weighted", zero_division=0)
    )

    f1_macro = float(
        f1_score(target_gather, preds, average="macro", zero_division=0)
    )

    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        target_gather,
        preds,
        average="macro",
        zero_division=0
    )

    balanced_acc = float(
        recall_score(target_gather, preds, average="macro", zero_division=0)
    )

    
    # Per-class metrics
    labels = np.unique(target_gather)

    precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
        target_gather,
        preds,
        labels=labels,
        zero_division=0
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
            roc_auc_score(
                y_1h,
                p,
                average="macro",
                multi_class="ovr"
            )
        )

        auprc = float(
            average_precision_score(
                y_1h,
                p,
                average="macro"
            )
        )

    cm = confusion_matrix(target_gather, preds, labels=labels)

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

def evaluate(model, criterion, loader, device):

    model.eval()

    total_loss = 0
    pred_gather = []
    target_gather = []

    with torch.no_grad():

        for embed_mri, embed_histo, modality_mask, target in loader:

            embed_mri = embed_mri.to(device)
            embed_histo = embed_histo.to(device)
            modality_mask = modality_mask.to(device)
            target = target.to(device)

            # forward pass (MASKED)
            output = model(
                embed_mri=embed_mri,
                embed_histo=embed_histo,
                mask=modality_mask
            )

            loss = criterion(output, target)
            total_loss += loss.item()

            pred_gather.append(output.detach().cpu().numpy())
            target_gather.append(target.detach().cpu().numpy())

    
    # Concatenate
    pred_gather = np.concatenate(pred_gather)
    target_gather = np.concatenate(target_gather)

    preds = pred_gather.argmax(1)

    total_samples = len(target_gather)
    avg_loss = total_loss / len(loader)

    
    # Global Metrics
    accuracy = float((preds == target_gather).mean())

    f1_weighted = float(
        f1_score(target_gather, preds, average="weighted", zero_division=0)
    )

    f1_macro = float(
        f1_score(target_gather, preds, average="macro", zero_division=0)
    )

    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        target_gather,
        preds,
        average="macro",
        zero_division=0
    )

    balanced_acc = float(
        recall_score(target_gather, preds, average="macro", zero_division=0)
    )

    
    # Per-class metrics
    labels = np.unique(target_gather)

    precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
        target_gather,
        preds,
        labels=labels,
        zero_division=0
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
            roc_auc_score(
                y_1h,
                p,
                average="macro",
                multi_class="ovr"
            )
        )

        auprc = float(
            average_precision_score(
                y_1h,
                p,
                average="macro"
            )
        )

    
    # Confusion Matrix
    cm = confusion_matrix(target_gather, preds, labels=labels)

    
    # Structured output
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
    # labels = np.array([dataset[i][2] if len(dataset[i]) == 3 else dataset[i][1] 
    #                    for i in range(len(dataset))])
    labels = np.array([dataset.label_dict[l] for l in dataset.labels])
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

def check_zero_rates(loader):
    total = 0
    zero_mri = 0
    zero_histo = 0
    both_zero = 0
    expected = 0
    # for embed_mri, embed_histo, modality_mask, target in loader:
    for embed_mri, embed_histo, modality_mask, target in loader:
        total += embed_mri.size(0)

        mri_is_zero = (embed_mri.abs().sum(dim=1) == 0)
        histo_is_zero = (embed_histo.abs().sum(dim=1) == 0)

        zero_mri += mri_is_zero.sum().item()
        zero_histo += histo_is_zero.sum().item()
        both_zero += (mri_is_zero & histo_is_zero).sum().item()

        expected += 2-modality_mask.sum()

    print("Zero MRI:", zero_mri, "/", total)
    print("Zero Histo:", zero_histo, "/", total)
    print("Both zero:", both_zero, "/", total)


def main():
    args = argparser.parse_args()
    print(args)

    # set the random seed
    seed_torch(torch.device('cuda'), args.seed)
    # set the processor
    processor_mri = ProcessorMRI()
    processor_histo = ProcessorHistopathology()
    # load the dataset
    splits = ['train', 'val', 'test']

    train_dataset, val_dataset, test_dataset = [MultimodalEmbeddingDataset(dataset_csv=args.dataset_csv, \
                                                zip_path_mri=args.zip_path_mri, zip_path_histo=args.zip_path_histo, split=split, \
                                                z_score=args.z_score, processor_mri=processor_mri, processor_histo=processor_histo) \
                                                for split in splits]




    # set num_classes
    args.num_classes = len(train_dataset.label_dict)
    summarize_dataset("Train", train_dataset)
    summarize_dataset("Val", val_dataset)
    summarize_dataset("Test", test_dataset)
    # print(f'Train: {len(train_dataset)}\tVal: {len(val_dataset)}\tTest: {len(test_dataset)}')

    # # infinite sampler for training
    # train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset, replacement=True)

    print("First 5 raw labels:")
    for i in range(5):
        label = train_dataset[i][3]
        print(i, type(label), label)

    # get training labels
    # train_labels = np.array([train_dataset[i][2] for i in range(len(train_dataset))])
    train_labels = np.array([train_dataset[i][3] for i in range(len(train_dataset))])
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts

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

    check_zero_rates(train_loader)
    check_zero_rates(test_loader)
   
 
    model = RobustGatedFusion(mri_probe_path=args.mri_probe_path,
                histo_probe_path=args.histo_probe_path, 
                 residual_hidden=64, freeze_probes=True, device="cuda")


    # Train the model
    train(model, train_loader, val_loader, test_loader, **vars(args))



class MultimodalEmbeddingDataset(Dataset):
    def __init__(self, dataset_csv, zip_path_mri, zip_path_histo,
                 split="train", z_score=False, processor_mri=None, processor_histo=None):

        df = pd.read_csv(dataset_csv)
        df = df[df["split"] == split].copy()
        df = df[df["label"].notna()]

        df["trial_accession"] = (
            df["trial_accession"]
            .astype(str)
            .str.strip()
            .str.replace(".0", "", regex=False)
        )

        self.mri_keys = df["trial_accession"].tolist()
        self.histo_keys = df["slide_id"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()

        self.embeds_mri = ProcessorMRI().load_embeddings_from_zip(zip_path_mri)
        self.embeds_histo = ProcessorHistopathology().load_embeddings_from_zip(zip_path_histo)

        self.mri_dim = next(iter(self.embeds_mri.values())).shape
        self.histo_dim = next(iter(self.embeds_histo.values())).shape

        all_labels = sorted(pd.read_csv(dataset_csv)["label"].dropna().unique())
        self.label_dict = {l: i for i, l in enumerate(all_labels)}

        self.z_score = z_score

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        mri_key = str(self.mri_keys[idx])
        histo_key = str(self.histo_keys[idx])
        target = self.label_dict[self.labels[idx]]

        # MRI
        if mri_key in self.embeds_mri:
            embed_mri = self.embeds_mri[mri_key].float()
            mri_mask = 1.0
        else:
            embed_mri = torch.zeros(self.mri_dim)
            mri_mask = 0.0

        # Histo
        if histo_key in self.embeds_histo:
            embed_histo = self.embeds_histo[histo_key].float()
            histo_mask = 1.0
        else:
            embed_histo = torch.zeros(self.histo_dim)
            histo_mask = 0.0

        if self.z_score:
            if mri_mask:
                embed_mri = (embed_mri - embed_mri.mean()) / (embed_mri.std() + 1e-8)
            if histo_mask:
                embed_histo = (embed_histo - embed_histo.mean()) / (embed_histo.std() + 1e-8)

        mask = torch.tensor([mri_mask, histo_mask], dtype=torch.float32)

        return embed_mri, embed_histo, mask, target



class ProcessorMRI:

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


class ProcessorHistopathology:

    def __init__(self, aggregation="mean"):
        # mean: Test Accuracy: 0.7857142857142857 f1: 0.6914285714285714 Precision: 0.39285714285714285 Recall: 0.5 AUROC: 0.7575757575757576 AUPRC: 0.7143562245834973
        # median: Test Accuracy: 0.7857142857142857 f1: 0.6914285714285714 Precision: 0.39285714285714285 Recall: 0.5 AUROC: 0.6818181818181819 AUPRC: 0.631065651520197
        assert aggregation in ["mean", "median"]
        self.aggregation = aggregation

    def load_embeddings_from_zip(self, zip_path, split=None):
        study_to_tensors = defaultdict(list)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in tqdm(zip_ref.infolist()):
                if not file_info.filename.endswith('.pt'):
                    continue

                # load tensor
                file_bytes = zip_ref.read(file_info.filename)
                tensor = torch.load(io.BytesIO(file_bytes))['last_layer_embed']

                # C013_A1.5_HE.pt -> C013
                sample_name = os.path.basename(file_info.filename).replace('.pt', '')
                study_code = sample_name.split('_')[0]

                study_to_tensors[study_code].append(tensor)

        # aggregate per study
        aggregated = {}
        for study_code, tensors in tqdm(study_to_tensors.items(), colour='red', total=len(study_to_tensors.items()
        )):
            aggregated[study_code] = {}
            stacked = torch.stack(tensors, dim=0)

            if self.aggregation == "mean":
                aggregated[study_code] = stacked.mean(dim=0)
            else:
                aggregated[study_code] = stacked.median(dim=0).values

        return aggregated




if __name__ == '__main__':
    main()

    # processor_mri, processor_histo = ProcessorMRI(), ProcessorHistopathology()
    # dataset_csv='/gscratch/kurtlab/CoreBT/dataset_utils/fusion/corebt_fusion_dataset_WHOGrade_case.csv'
    # zip_path_mri='/gscratch/scrubbed/juampablo/corebt/mri_embeddings.zip'
    # zip_path_histo='/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/slide_embeddings.zip'
    # ds = MultimodalEmbeddingDataset(dataset_csv=dataset_csv, zip_path_mri=zip_path_mri, zip_path_histo=zip_path_histo, split = 'train', z_score=False, processor_mri=processor_mri, processor_histo=processor_histo)

    # EmbeddingDataset(dataset_csv, zip_path, \
    #                 split='train', z_score=False, processor=processor)
    # EMBED_DIM=1536


# DATASET_CSV=/gscratch/kurtlab/CoreBT/dataset_utils/fusion/corebt_fusion_dataset_WHOGrade_case.csv
# ZIP_PATH_MRI=/gscratch/scrubbed/juampablo/corebt/mri_embeddings.zip
# ZIP_PATH_HISTO=/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/slide_embeddings.zip
# OUTPUT_DIR=/gscratch/kurtlab/CoreBT/fusion/experiments/corebt_linear_probe
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

# python3 -m corebt_fusion_main --dataset_csv $DATASET_CSV --zip_path_mri $ZIP_PATH_MRI --zip_path_histo $ZIP_PATH_HISTO --batch_size $BATCH_SIZE --train_iters $TRAIN_ITERS --lr $LR --min_lr $MIN_LR --optim $OPTIM --momentum $MOMENTUM --weight_decay $WEIGHT_DECAY --eval_interval $EVAL_INTERVAL --num_workers $NUM_WORKERS --seed $SEED --output_dir $OUTPUT_DIR