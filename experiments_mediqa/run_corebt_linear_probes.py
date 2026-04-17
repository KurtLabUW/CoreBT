import argparse, json, subprocess
from pathlib import Path
from datetime import datetime


COMMON_ARGS = {
    "--batch_size": 32,
    # "--train_iters": 1200,
    "--train_iters": 600,
    "--lr": 0.0001, 
    "--min_lr": 0.0,
    "--optim": "adam",
    "--momentum": 0.0,
    "--weight_decay": 1e-4,
    "--eval_interval": 10,
    "--num_workers": 4,
    "--seed": 42,
}

def build_mri_config(args):
    output_dir = Path(args.output_dir) / args.run_name
    if args.label_prefix != "all":
        output_dir /= args.label_prefix
    return {
        "name": "mri_only",
        "command": [
            "uv", "run", "python",
            "corebt_mri_main.py",
        ],
        "args": {
            **COMMON_ARGS,
            "--train_csv_path": args.train_csv_path,
            "--val_csv_path": args.val_csv_path,
            "--test_csv_path": args.test_csv_path,
            "--train_mri_embed_path": args.train_mri_embed_path,
            "--val_mri_embed_path": args.val_mri_embed_path,
            "--test_mri_embed_path": args.test_mri_embed_path,
            "--output_dir": output_dir / 'mri',
            "--label_prefix": args.label_prefix,
            "--embed_dim": 768,
        },
    }


def build_histo_config(args):
    output_dir = Path(args.output_dir) / args.run_name
    if args.label_prefix != "all":
        output_dir /= args.label_prefix
    return {
        "name": "histo_only",
        "command": [
            "uv", "run", "python",
            "corebt_histo_main.py",
        ],
        "args": {
            **COMMON_ARGS,
            "--train_csv_path": args.train_csv_path,
            "--val_csv_path": args.val_csv_path,
            "--test_csv_path": args.test_csv_path,
            "--train_histo_embed_path": args.train_histo_embed_path,
            "--val_histo_embed_path": args.val_histo_embed_path,
            "--test_histo_embed_path": args.test_histo_embed_path,
            "--output_dir": output_dir / 'histopathology',
            "--label_prefix": args.label_prefix,
            "--embed_dim": 768,
        },
    }


def build_fusion_config(args):
    output_dir = Path(args.output_dir) / args.run_name
    if args.label_prefix != "all":
        output_dir /= args.label_prefix
    return {
        "name": "fusion_mri_histo",
        "command": [
            "uv", "run", "python",
            "corebt_fusion_main.py",
        ],
        "args": {
            **COMMON_ARGS,
            "--train_csv_path": args.train_csv_path,
            "--val_csv_path": args.val_csv_path,
            "--test_csv_path": args.test_csv_path,
            "--train_mri_embed_path": args.train_mri_embed_path,
            "--val_mri_embed_path": args.val_mri_embed_path,
            "--test_mri_embed_path": args.test_mri_embed_path,
            "--train_histo_embed_path": args.train_histo_embed_path,
            "--val_histo_embed_path": args.val_histo_embed_path,
            "--test_histo_embed_path": args.test_histo_embed_path,
            "--output_dir": output_dir / 'fusion',
            "--label_prefix": args.label_prefix,
        },
    }

def collect_results(args, configs, label_prefix):
    print("\nCollecting results...")

    shared_dir = Path('/gscratch/kurtlab/CoreBT/experiments_mediqa/') / args.run_name / label_prefix
    shared_dir.mkdir(parents=True, exist_ok=True)

    combined_results = {}

    for config in configs:
        variant_name = config["name"]
        output_dir = Path(config["args"]["--output_dir"])

        result_json = output_dir / "results.json"

        if result_json.exists():
            with open(result_json, "r") as f:
                combined_results[variant_name] = json.load(f)
            print(f"Collected {variant_name}")
        else:
            print(f"Warning: No results.json found for {variant_name}")

    summary_path = shared_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(combined_results, f, indent=4)

    print(f"\nSaved combined summary to:\n{summary_path}")



def build_command(config):
    cmd = config["command"].copy()
    for k, v in config["args"].items():
        cmd.append(k)
        cmd.append(str(v))
    return cmd


def run_variant(config):
    print("\n" + ":" * 80)
    print(f"Running variant: {config['name']}")
    print(":" * 80)

    cmd = build_command(config)
    print("Command:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)



def get_next_run_id(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")

    existing = [
        p.name for p in output_dir.iterdir()
        if p.is_dir() and today in p.name
    ]

    if not existing:
        idx = 1
    else:
        nums = []
        for name in existing:
            try:
                nums.append(int(name.split("_r")[-1]))
            except:
                pass
        idx = max(nums) + 1 if nums else 1

    return f"{today}_r{idx:03d}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv_path", type=str, required=True)
    parser.add_argument("--val_csv_path", type=str, required=True)
    parser.add_argument("--test_csv_path", type=str, required=True)

    parser.add_argument("--train_mri_embed_path", type=str, required=True)
    parser.add_argument("--val_mri_embed_path", type=str, required=True)
    parser.add_argument("--test_mri_embed_path", type=str, required=True)

    parser.add_argument("--train_histo_embed_path", type=str, required=True)
    parser.add_argument("--val_histo_embed_path", type=str, required=True)
    parser.add_argument("--test_histo_embed_path", type=str, required=True)


    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument(
        "--variant",
        type=str,
        choices=["mri", "histo", "fusion", "all"],
        default="all",
    )

    parser.add_argument(
        "--label_prefix",
        type=str,
        choices=["level1", "lgghgg", "who_grade", "all"],
        default="all",
    )

    args = parser.parse_args()

    run_id = get_next_run_id(args.output_dir)
    if args.run_name is None:
        args.run_name = f"run_{run_id}"

    configs_ran = []

    if args.variant in ["mri", "all"]:
        cfg = build_mri_config(args)
        run_variant(cfg)
        configs_ran.append(cfg)

    if args.variant in ["histo", "all"]:
        cfg = build_histo_config(args)
        run_variant(cfg)
        configs_ran.append(cfg)

    if args.variant in ["fusion", "all"]:
        cfg = build_fusion_config(args)
        run_variant(cfg)
        configs_ran.append(cfg)

    # collect_results(args, configs_ran)

    print("\nAll requested runs completed successfully.")



'''
CSV_DIR=/gscratch/kurtlab/CoreBT/experiments_mediqa/dataset_csvs
EMBED_DIR=/gscratch/scrubbed/juampablo/corebt_dataset
OUTPUT_DIR=/gscratch/kurtlab/CoreBT/experiments_mediqa/runs

python3 run_corebt_linear_probes.py \
    --train_csv_path "$CSV_DIR/train.csv" \
    --val_csv_path "$CSV_DIR/train.csv" \
    --test_csv_path "$CSV_DIR/val_randomized.csv" \
    --train_mri_embed_path "$EMBED_DIR/MRI_Embeddings_train.zip" \
    --val_mri_embed_path "$EMBED_DIR/MRI_Embeddings_train.zip" \
    --test_mri_embed_path "$EMBED_DIR/MRI_Embeddings_val.zip" \
    --train_histo_embed_path "$EMBED_DIR/Pathology_Embeddings_train.zip" \
    --val_histo_embed_path "$EMBED_DIR/Pathology_Embeddings_train.zip" \
    --test_histo_embed_path "$EMBED_DIR/Pathology_Embeddings_val.zip" \
    --output_dir "$OUTPUT_DIR" \
    --variant all \
    --label_prefix all
    
'''