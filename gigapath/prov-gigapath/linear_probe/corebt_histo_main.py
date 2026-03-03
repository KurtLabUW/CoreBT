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


'''
================================================================================
Validation Results
================================================================================
Samples: 21
Average Loss: 1.3282
Accuracy: 0.5714
Balanced Accuracy: 0.2308
F1 (macro): 0.1875
F1 (weighted): 0.4643
Precision (macro): 0.1579
Recall (macro): 0.2308
AUROC (macro): 0.8250
AUPRC (macro): 0.6678

Per-Class Metrics
--------------------------------------------------------------------------------
Class | Support | Fraction | Precision | Recall | F1
--------------------------------------------------------------------------------
    0 |       1 | 1/21 (  4.8%) |     0.000 |  0.000 | 0.000
    1 |       3 | 3/21 ( 14.3%) |     0.000 |  0.000 | 0.000
    2 |       4 | 4/21 ( 19.0%) |     0.000 |  0.000 | 0.000
    3 |      13 | 13/21 ( 61.9%) |     0.632 |  0.923 | 0.750

Confusion Matrix
--------------------------------------------------------------------------------
[[ 0  0  0  1]
 [ 0  0  1  2]
 [ 0  0  0  4]
 [ 0  1  0 12]]
================================================================================

Test Accuracy: 0.5714285714285714 f1: 0.4642857142857143 Precision: 0.15789473684210525 Recall: 0.23076923076923078 AUROC: 0.824995810289928 AUPRC: 0.6678127170387231

'''


def seed_torch(device, seed=7):
    # ------------------------------------------------------------------------------------------
    # References:
    # HIPT: https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/main.py
    # ------------------------------------------------------------------------------------------
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
    ----------
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
            results = evaluate(model, criterion, test_loader, device)

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

    # Append-friendly JSON (list of runs)
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
    
    # writer.add_scalar('Test Accuracy', accuracy, i)
    # writer.add_scalar('Test f1', f1, i)
    # writer.add_scalar('Test AUROC', auroc, i)
    # writer.add_scalar('Test AUPRC', auprc, i)
    # writer.add_scalar('Test Precision', precision, i)
    # writer.add_scalar('Test Recall', recall, i)

    # f = open(f'{output_dir}/results.txt', 'w')
    # f.write(f'Val f1: {val_f1}\n')
    # f.write(f'Test f1: {f1} Test AUROC: {auroc} Test AUPRC: {auprc}\n')
    # f.close()


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

    # -----------------------
    # Global Metrics
    # -----------------------
    accuracy = float((preds == target_gather).mean())
    f1_weighted = float(f1_score(target_gather, preds, average="weighted"))
    f1_macro = float(f1_score(target_gather, preds, average="macro"))

    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        target_gather, preds, average="macro", zero_division=0
    )

    balanced_acc = float(
        recall_score(target_gather, preds, average="macro", zero_division=0)
    )

    # -----------------------
    # Per-class metrics
    # -----------------------
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

    # -----------------------
    # AUROC / AUPRC
    # -----------------------
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

    # -----------------------
    # Confusion Matrix
    # -----------------------
    cm = confusion_matrix(target_gather, preds, labels=labels)

    # -----------------------
    # Assemble structured results
    # -----------------------
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

# def evaluate(model, criterion, val_loader, device):
#     model.eval()

#     total_loss = 0
#     pred_gather, target_gather = [], []

#     with torch.no_grad():
#         for embed, target in val_loader:

#             embed = embed.to(device)
#             target = target.to(device)

#             output = model(embed)
#             loss = criterion(output, target)
#             total_loss += loss.item()

#             pred_gather.append(output.cpu().numpy())
#             target_gather.append(target.cpu().numpy())

#     pred_gather = np.concatenate(pred_gather)
#     target_gather = np.concatenate(target_gather)

#     preds = pred_gather.argmax(1)

#     # -----------------------
#     # Global Metrics
#     # -----------------------
#     accuracy = (preds == target_gather).mean()
#     f1_weighted = f1_score(target_gather, preds, average="weighted")
#     f1_macro = f1_score(target_gather, preds, average="macro")
#     precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
#         target_gather, preds, average="macro", zero_division=0
#     )

#     balanced_acc = recall_score(target_gather, preds, average="macro", zero_division=0)

#     avg_loss = total_loss / len(val_loader)

#     # -----------------------
#     # Per-class metrics
#     # -----------------------
#     labels = np.unique(target_gather)
#     precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
#         target_gather, preds, labels=labels, zero_division=0
#     )

#     total_samples = len(target_gather)

#     # -----------------------
#     # AUROC / AUPRC
#     # -----------------------
#     present = np.unique(target_gather.astype(int))

#     if present.size < 2:
#         auroc = float("nan")
#         auprc = float("nan")
#     else:
#         y_1h = to_onehot(target_gather, pred_gather.shape[1])
#         y_1h = y_1h[:, present]
#         p = pred_gather[:, present]

#         auroc = roc_auc_score(y_1h, p, average="macro", multi_class="ovr")
#         auprc = average_precision_score(y_1h, p, average="macro")

#     # -----------------------
#     # Printing
#     # -----------------------
#     print("\n" + "="*80)
#     print("Validation Results")
#     print("="*80)

#     print(f"Samples: {total_samples}")
#     print(f"Average Loss: {avg_loss:.4f}")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Balanced Accuracy: {balanced_acc:.4f}")
#     print(f"F1 (macro): {f1_macro:.4f}")
#     print(f"F1 (weighted): {f1_weighted:.4f}")
#     print(f"Precision (macro): {precision_macro:.4f}")
#     print(f"Recall (macro): {recall_macro:.4f}")
#     print(f"AUROC (macro): {auroc:.4f}")
#     print(f"AUPRC (macro): {auprc:.4f}")

#     print("\nPer-Class Metrics")
#     print("-"*80)
#     print("Class | Support | Fraction | Precision | Recall | F1")
#     print("-"*80)

#     for i, cls in enumerate(labels):
#         frac = support_c[i] / total_samples
#         print(
#             f"{cls:5d} | "
#             f"{support_c[i]:7d} | "
#             f"{support_c[i]}/{total_samples} ({frac*100:5.1f}%) | "
#             f"{precision_c[i]:9.3f} | "
#             f"{recall_c[i]:6.3f} | "
#             f"{f1_c[i]:5.3f}"
#         )

#     # -----------------------
#     # Confusion Matrix
#     # -----------------------
#     cm = confusion_matrix(target_gather, preds, labels=labels)

#     print("\nConfusion Matrix")
#     print("-"*80)
#     print(cm)

#     print("="*80 + "\n")

#     return accuracy, f1_weighted, precision_macro, recall_macro, auroc, auprc



def main():
    args = argparser.parse_args()
    print(args)

    # set the random seed
    seed_torch(torch.device('cuda'), args.seed)
    # set the processor
    processor = ProcessorSubject()
    # load the dataset
    splits = ['train', 'val', 'test']
    train_dataset, val_dataset, test_dataset = [EmbeddingDataset(args.dataset_csv, args.input_path, \
                    split=split, z_score=args.z_score, processor=processor) for split in splits]
    # set num_classes
    args.num_classes = len(train_dataset.label_dict)
    print(f'Train: {len(train_dataset)}\tVal: {len(val_dataset)}\tTest: {len(test_dataset)}')

    # infinite sampler for training
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset, replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Load the model
    model = LinearProbe(args.embed_dim, args.num_classes)

    # Train the model
    train(model, train_loader, val_loader, test_loader, **vars(args))


class LinearProbe(nn.Module):

    def __init__(self, embed_dim: int = 1536, num_classes: int = 10):
        super(LinearProbe, self).__init__()

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class EmbeddingDataset(Dataset):
    def __init__(self, dataset_csv: str, zip_path: str, split: str = 'train', z_score=False, processor=None):
        """
        Dataset used for training the linear probe based on the embeddings extracted from the pre-trained model.

        Arugments:
        dataset_csv (str): Path to the csv file containing the embeddings and labels.
        """
        df = pd.read_csv(dataset_csv)
        split_df = df[df['split'] == split]


        # self.samples = split_df['slide_id'].tolist()
        # # self.samples = split_df['input'].tolist()
        # self.labels = split_df['label'].tolist()


        # Remove rows with NaN labels
        df = split_df.copy()

        df = df[df["label"].notna()]
        df = df[df["slide_id"].notna()]
        df = df[df["slide_id"].astype(str).str.strip() != ""]
        df = df[df["slide_id"].astype(str).str.lower() != "nan"]


        # Keep only subjects with histopathology present
        if "histo_present" in df.columns:
            df = df[df["histo_present"] == True]

            print(
                f"[Dataset Init] Using {len(df)} subjects "
                f"with histo_present == True and non-NaN labels "
                f"(filtered from {len(split_df)})"
            )
        else:
            print(
                f"[Dataset Init] histo_present column not found. "
                f"Using {len(df)} subjects with non-NaN labels."
            )

        # Assign samples and labels from filtered dataframe
        self.samples = df["slide_id"].astype(str).tolist()
        self.labels = df["label"].tolist()


        # load the embeddings
        self.processor = processor
        self.embeds = processor.load_embeddings_from_zip(zip_path, split)

        # label_set = list(set(self.labels))
        # label_set.sort()
        # self.label_dict = {label: i for i, label in enumerate(label_set)}
        all_labels = sorted(pd.read_csv(dataset_csv)['label'].unique())
        self.label_dict = {label: i for i, label in enumerate(all_labels)}
        
        self.z_score = z_score

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample, target = self.samples[index], self.labels[index]
        embed = self.embeds[sample]['last_layer_embed']

        if self.z_score:
            # z-score normalization
            embed = (embed - embed.mean()) / embed.std()

        # convert the label to index
        target = self.label_dict[target]

        return embed, target


class ProcessorSubject:

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
                aggregated[study_code]['last_layer_embed'] = stacked.mean(dim=0)
            else:
                aggregated[study_code]['last_layer_embed'] = stacked.median(dim=0).values

        return aggregated



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
        return loaded_tensors


if __name__ == '__main__':
    main()
    # processor = Processor()
    # dataset_csv='/gscratch/kurtlab/CoreBT/dataset_utils/corebt_patho_dataset_WHOGrade.csv'
    # zip_path= '/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/slide_embeddings.zip'
    # EmbeddingDataset(dataset_csv, zip_path, \
    #                 split='train', z_score=False, processor=processor)
# EMBED_DIM=1536


# DATASET_CSV=/gscratch/kurtlab/CoreBT/dataset_utils/corebt_patho_dataset_WHOGrade.csv
# ZIP_PATH=/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/slide_embeddings.zip
# OUTPUT_DIR=/gscratch/kurtlab/CoreBT/experiments/corebt_linear_probe
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

# python3 -m linear_probe.corebt_main --dataset_csv $DATASET_CSV --input_path $ZIP_PATH --embed_dim $EMBED_DIM --batch_size $BATCH_SIZE --train_iters $TRAIN_ITERS --lr $LR --min_lr $MIN_LR --optim $OPTIM --momentum $MOMENTUM --weight_decay $WEIGHT_DECAY --eval_interval $EVAL_INTERVAL --num_workers $NUM_WORKERS --seed $SEED --output_dir $OUTPUT_DIR