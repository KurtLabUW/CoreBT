import argparse, json, subprocess
from pathlib import Path


COMMON_ARGS = {
    "--batch_size": 32,
    "--train_iters": 1200,
    "--lr": 0.001,
    "--min_lr": 0.0,
    "--optim": "adam",
    "--momentum": 0.0,
    "--weight_decay": 1e-4,
    "--eval_interval": 200,
    "--num_workers": 4,
    "--seed": 42,
}

def build_mri_config(dataset_csv, run_name):
    return {
        "name": "mri_only",
        "command": [
            "python3",
            "/gscratch/kurtlab/CoreBT/experiments/corebt_mri_main.py",
        ],
        "args": {
            **COMMON_ARGS,
            "--dataset_csv": dataset_csv,
            "--input_path": "/gscratch/scrubbed/juampablo/corebt/mri_embeddings.zip",
            "--output_dir": f"/gscratch/kurtlab/CoreBT/experiments/corebt_linear_probe_v2/{run_name}/mri_only",
            "--embed_dim": 768,
        },
    }


def build_histo_config(dataset_csv, run_name):
    return {
        "name": "histo_only",
        "command": [
            "python3",
            "/gscratch/kurtlab/CoreBT/experiments/corebt_histo_main.py",
        ],
        "args": {
            **COMMON_ARGS,
            "--dataset_csv": dataset_csv,
            "--input_path": "/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/slide_embeddings.zip",
            "--output_dir": f"/gscratch/kurtlab/CoreBT/experiments/corebt_linear_probe_v2/{run_name}/histo_only",
            "--embed_dim": 768,
        },
    }


def build_fusion_config(dataset_csv, run_name):
    return {
        "name": "fusion_mri_histo",
        "command": [
            "python3",
            "/gscratch/kurtlab/CoreBT/experiments/corebt_fusion_main.py",
        ],
        "args": {
            **COMMON_ARGS,
            "--dataset_csv": dataset_csv,
            "--zip_path_mri": "/gscratch/scrubbed/juampablo/corebt/mri_embeddings.zip",
            "--zip_path_histo": "/gscratch/scrubbed/juampablo/corebt/corebt_clam_preprocessing/slide_embeddings.zip",
            "--output_dir": f"/gscratch/kurtlab/CoreBT/experiments/corebt_linear_probe_v2/{run_name}/fusion_mri_histo/",
            "--mri_probe_path": f"/gscratch/kurtlab/CoreBT/experiments/corebt_linear_probe_v2/{run_name}/mri_only/best_model.pth",
            "--histo_probe_path": f"/gscratch/kurtlab/CoreBT/experiments/corebt_linear_probe_v2/{run_name}/histo_only/best_model.pth",
        },
    }

def collect_results(run_name, configs):
    print("\nCollecting results...")

    shared_dir = Path('/gscratch/kurtlab/CoreBT/experiments/corebt_linear_probe_v2/') / run_name
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


'''
python3 run_corebt_linear_probes.py --dataset_csv /gscratch/kurtlab/CoreBT/dataset_utils/fusion/corebt_fusion_dataset_WHOGrade_case.csv --run_name WHOGrade
python3 run_corebt_linear_probes.py --dataset_csv /gscratch/kurtlab/CoreBT/dataset_utils/fusion/corebt_fusion_dataset_LGGHGG_case.csv --run_name LGG_HGG
python3 run_corebt_linear_probes.py --dataset_csv /gscratch/kurtlab/CoreBT/dataset_utils/fusion/corebt_fusion_dataset_Level1_case.csv --run_name Level1Class


python3 run_corebt_linear_probes.py --dataset_csv /gscratch/kurtlab/CoreBT/dataset_utils/fusion/corebt_fusion_dataset_WHOGrade_case_sharedtest.csv --run_name WHOGrade
python3 run_corebt_linear_probes.py --dataset_csv /gscratch/kurtlab/CoreBT/dataset_utils/fusion/corebt_fusion_dataset_LGG_HGG_case_sharedtest.csv --run_name LGG_HGG
python3 run_corebt_linear_probes.py --dataset_csv /gscratch/kurtlab/CoreBT/dataset_utils/fusion/corebt_fusion_dataset_LEVEL1_case_sharedtest.csv --run_name Level1Class


'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)

    parser.add_argument(
        "--variant",
        type=str,
        choices=["mri", "histo", "fusion", "all"],
        default="all",
    )

    args = parser.parse_args()

    configs_ran = []

    if args.variant in ["mri", "all"]:
        cfg = build_mri_config(args.dataset_csv, args.run_name)
        run_variant(cfg)
        configs_ran.append(cfg)

    if args.variant in ["histo", "all"]:
        cfg = build_histo_config(args.dataset_csv, args.run_name)
        run_variant(cfg)
        configs_ran.append(cfg)

    if args.variant in ["fusion", "all"]:
        cfg = build_fusion_config(args.dataset_csv, args.run_name)
        run_variant(cfg)
        configs_ran.append(cfg)

    collect_results(args.run_name, configs_ran)

    print("\nAll requested runs completed successfully.")