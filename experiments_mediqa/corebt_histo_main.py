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
import plotext as plt


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
# argparser.add_argument('--dataset_csv_dir',            type=str, default='', help='The dir containing csv files indicating input samples and labels')
# argparser.add_argument('--embeddings_zip_dir',          type=str, default='', help='The input embedding files')
argparser.add_argument('--embed_dim',           type=int, default=1536, help='The dimension of the embeddings')
# argparser.add_argument('--label_prefix',           type=str, default=None, help='The prefix to the _label column in the csv. Defaults to None, which just searches for label in the csv')
argparser.add_argument(
        "--label_prefix",
        type=str,
        choices=["level1", "lgghgg", "who_grade", "all"],
        default="all",
    )

# Training
argparser.add_argument('--batch_size',          type=int, default=512, help='Batch size')
argparser.add_argument('--train_iters',         type=int, default=12500, help='Number of steps')
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


# Metadata CSV Paths
argparser.add_argument('--train_csv_path', type=str, default='dataset_csvs/train.csv', help='Full path to training CSV')
argparser.add_argument('--val_csv_path',   type=str, default='dataset_csvs/val.csv', help='Full path to validation CSV')
argparser.add_argument('--test_csv_path',  type=str, default='dataset_csvs/test.csv', help='Full path to test/dummy CSV')

# MRI Embedding Paths
argparser.add_argument('--train_mri_embed_path', type=str, default='corebt_dataset/MRI_Embeddings_train.zip', help='Full path to training MRI zip')
argparser.add_argument('--val_mri_embed_path',   type=str, default='corebt_dataset/MRI_Embeddings_val.zip', help='Full path to validation MRI zip')
argparser.add_argument('--test_mri_embed_path',  type=str, default='corebt_dataset/MRI_Embeddings_test.zip', help='Full path to test MRI zip')

# Histopathology Embedding Paths
argparser.add_argument('--train_histo_embed_path', type=str, default='corebt_dataset/Pathology_Embeddings_train.zip', help='Full path to training Pathology zip')
argparser.add_argument('--val_histo_embed_path',   type=str, default='corebt_dataset/Pathology_Embeddings_val.zip', help='Full path to validation Pathology zip')
argparser.add_argument('--test_histo_embed_path',  type=str, default='corebt_dataset/Pathology_Embeddings_test.zip', help='Full path to test Pathology zip')




def to_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    '''Convert the labels to one-hot encoding'''
    onehot = np.zeros((labels.shape[0], num_classes))
    onehot[np.arange(labels.shape[0]), labels] = 1
    return onehot


def train(model,
          train_loader,
          val_loader,
          test_loader,
          criterion,
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
    # criterion = nn.CrossEntropyLoss()


    # Set the infinite train loader
    infinite_train_loader = itertools.cycle(train_loader)

    best_f1 = 0
    # Train the model
    print('Start training')
    train_losses = []
    loss_steps = []

    val_f1s = []
    val_aurocs = []
    val_steps = []
    for i, (embed, target, subject_ids) in enumerate(infinite_train_loader):

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
            results, predictions = evaluate(model, criterion, val_loader, device)

            accuracy = results["global_metrics"]["accuracy"]
            f1 = results["global_metrics"]["f1_weighted"]
            precision = results["global_metrics"]["precision_macro"]
            recall = results["global_metrics"]["recall_macro"]
            auroc = results["global_metrics"]["auroc_macro"]
            auprc = results["global_metrics"]["auprc_macro"]          
            
            last_f1 = f1
            val_f1s.append(f1)
            val_steps.append(i + 1)

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


    plt.clf()                     # clear previous plots
    plt.plotsize(100, 25)         # width, height in terminal chars
    plt.title("Val F1/AUROC vs iter")
    plt.xlabel("Iteration")
    plt.ylabel("Value")

    plt.plot(val_steps, val_f1s, label='Val F1')
    plt.plot(val_steps, val_aurocs, label='Val AUROC')

    try:
        plt.show()
        plt.savefig(f'{output_dir}/metrics_plot.py')

    except Exception as e:
        print(f"Plotext failed to render: {e}")
        


    if kwargs.get('model_select') == 'best':
        val_f1 = best_f1
        model.load_state_dict(torch.load(f'{output_dir}/best_model.pth'))
    else:
        val_f1 = f1
        model.load_state_dict(torch.load(f'{output_dir}/model.pth'))

    # Evaluate the model
    results, predictions = evaluate(model, criterion, test_loader, device)
    results_df = pd.DataFrame(predictions)

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
    
    return results_df


def evaluate(model, criterion, val_loader, device):
    model.eval()

    total_loss = 0
    pred_gather, target_gather = [], []
    predictions = []

    with torch.no_grad():
        for embed, target, subject_ids in val_loader:

            embed = embed.to(device)
            target = target.to(device)

            output = model(embed)
            loss = criterion(output, target)
            total_loss += loss.item()

            pred_gather.append(output.cpu().numpy())
            target_gather.append(target.cpu().numpy())
                
            for sid, pred, targ in zip(subject_ids, output.cpu().numpy(), target.cpu().numpy()):

                predictions.append({
                    'subject_id': sid,
                    'prediction': pred.argmax().item()
                })
                
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

    return results, predictions

def print_split_stats(name, dataset):
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n{name} Split Label Distribution:")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c}")
    print("Total:", len(labels))

def main():
    args = argparser.parse_args()
    print(args)
    
    # args.output_dir = os.path.join(args.output_dir, args.label_prefix,  'histopathology')
    # set the random seed
    seed_torch(torch.device('cuda'), args.seed)

    all_prefix_tasks = ["level1", "lgghgg", "who_grade"]    
    if args.label_prefix == "all":
        tasks_to_execute = all_prefix_tasks
    else:
        tasks_to_execute = [args.label_prefix]

    # set the processor
    processor = ProcessorSubject()

    # load the dataset
    csv_paths = [args.train_csv_path, args.val_csv_path, args.test_csv_path]
    histo_paths = [args.train_histo_embed_path, args.val_histo_embed_path, args.test_histo_embed_path]

    train_dataset, val_dataset, test_dataset = [
        EmbeddingDataset(
            dataset_csv=csv_paths[ix],
            zip_path=histo_paths[ix],
            label_prefix=args.label_prefix,
            z_score=args.z_score,
            processor=processor,
        ) for ix in range(3)
    ]


    base_output_dir = args.output_dir
    for prefix in tasks_to_execute:

        args.output_dir = os.path.join(base_output_dir, prefix)

        train_dataset.update_labels(prefix)
        val_dataset.update_labels(prefix)
        test_dataset.update_labels(prefix)

        # set num_classes
        args.num_classes = len(train_dataset.label_dict)

        print(f'Task: {prefix} ---- Classes: {args.num_classes}')
        print_split_stats("Train", train_dataset)
        print_split_stats("Val", val_dataset)
        print_split_stats("Test", test_dataset)

        # infinite sampler for training
        # train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset, replacement=True)

        # get training labels
        # train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
        # train_labels = np.array(train_dataset.labels)

        train_labels = np.array(train_dataset.labels)

        class_counts = np.bincount(
            train_labels,
            minlength=args.num_classes
        )
        # class_counts = np.bincount(train_labels)
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

        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to('cuda:0')
        criterion = nn.CrossEntropyLoss()


        # Train the model
        results_df = train(model, train_loader, val_loader, test_loader, criterion, **vars(args))
        pred_col = f"{prefix}_pred"
        results_df.rename(columns={'prediction': pred_col}, inplace=True)

        results_df_savepath=os.path.join(args.output_dir, 'test_predictions.csv')
        os.makedirs(os.path.dirname(results_df_savepath), exist_ok=True)

        results_df = results_df[['subject_id', pred_col]]

        if os.path.exists(results_df_savepath):
            existing_df = pd.read_csv(results_df_savepath).set_index('subject_id')
            results_df = results_df.set_index('subject_id')

            existing_df[pred_col] = results_df[pred_col]

            merged_df = existing_df.reset_index()
        else:
            merged_df = results_df

        merged_df.to_csv(results_df_savepath, index=False)




class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int = 768, num_classes: int = 10):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.ln(x)
        return self.fc(x)


class EmbeddingDataset(Dataset):
    def __init__(self, dataset_csv: str, zip_path: str, split: str = 'train', z_score=False, processor=None, label_prefix=None):
        """
        Dataset used for training the linear probe based on the embeddings extracted from the pre-trained model.

        Arugments:
        dataset_csv (str): Path to the csv file containing the embeddings and labels.
        """
        split_df = pd.read_csv(dataset_csv, dtype={"subject_id": str})
        label_key="label" if not label_prefix else f"{label_prefix}_label"
        self.z_score = z_score

        # filter out mri only
        if "histopathology_present" in split_df.columns:
            split_df = split_df[split_df["histopathology_present"] == True].copy()

        # load the embeddings
        self.processor = processor
        self.embeds = processor.load_embeddings_from_zip(zip_path)
        
        embed_keys = set(sorted(self.embeds.keys()))
        # print(f'Subject keys: {set(sorted(split_df["subject_id"].tolist()))}')
        # print(f'EMbed keys: {embed_keys}')
        # before = len(split_df)
        self.split_df = split_df[split_df["subject_id"].isin(embed_keys)].copy()
        self.samples = self.split_df["subject_id"].astype(str).tolist()

        self.labels = []
        self.label_dict = {}

        if label_prefix and label_prefix != "all":
            self.update_labels(label_prefix)


    def update_labels(self, label_prefix):
        """
        Call this method to swap the task (e.g., from 'who_grade' to 'lgghgg') without reloading the embedding tensors from the zip.
        """
        label_key = f"{label_prefix}_label"
        
        label_prefix_to_all_labels_mapping = {
            'who_grade': [0, 1, 2],
            'level1':    [0, 1, 2, 3],
            'lgghgg':    [0, 1],
        }

        if label_prefix not in label_prefix_to_all_labels_mapping:
            raise ValueError(
                f"Invalid label_prefix: '{label_prefix}'. "
                f"Must be one of {list(label_prefix_to_all_labels_mapping.keys())}"
            )

        all_labels = label_prefix_to_all_labels_mapping[label_prefix]
        
        self.label_dict = {int(label): i for i, label in enumerate(all_labels)}
        self.labels = self.split_df[label_key].astype(int).tolist()      

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

        return embed, target, sample


class ProcessorSubject:

    def __init__(self, aggregation="mean"):
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
                study_code = sample_name.split('-WSI')[0]

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


# # Base Directories
# CSV_DIR=dataset_csvs
# EMBED_DIR=/gscratch/scrubbed/juampablo/corebt_dataset

# # Metadata CSV Paths
# TRAIN_CSV=$CSV_DIR/train.csv
# VAL_CSV=$CSV_DIR/train.csv          
# TEST_CSV=$CSV_DIR/val_randomized.csv

# # Histopathology Embedding Paths (Specific to this script)
# TRAIN_HISTO=$EMBED_DIR/Pathology_Embeddings_train.zip
# VAL_HISTO=$EMBED_DIR/Pathology_Embeddings_train.zip
# TEST_HISTO=$EMBED_DIR/Pathology_Embeddings_val.zip

# # Hyperparameters
# OUTPUT_DIR=run/histopathology
# EMBED_DIM=768
# BATCH_SIZE=32
# TRAIN_ITERS=400
# LR=0.001
# MIN_LR=0.0
# OPTIM=adam
# MOMENTUM=0.0
# WEIGHT_DECAY=1e-4
# EVAL_INTERVAL=200
# NUM_WORKERS=4
# SEED=42
# LABEL_PREFIX='all' # choices: level1, lgghgg, who_grade, all

# python3 -m corebt_histo_main \
#     --train_csv_path "$TRAIN_CSV" \
#     --val_csv_path "$VAL_CSV" \
#     --test_csv_path "$TEST_CSV" \
#     --train_histo_embed_path "$TRAIN_HISTO" \
#     --val_histo_embed_path "$VAL_HISTO" \
#     --test_histo_embed_path "$TEST_HISTO" \
#     --label_prefix $LABEL_PREFIX \
#     --embed_dim $EMBED_DIM \
#     --batch_size $BATCH_SIZE \
#     --train_iters $TRAIN_ITERS \
#     --lr $LR \
#     --min_lr $MIN_LR \
#     --optim $OPTIM \
#     --momentum $MOMENTUM \
#     --weight_decay $WEIGHT_DECAY \
#     --eval_interval $EVAL_INTERVAL \
#     --num_workers $NUM_WORKERS \
#     --seed $SEED \
#     --output_dir $OUTPUT_DIR