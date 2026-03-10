import os
import json
import pandas as pd

ROOT_DIR = "/gscratch/kurtlab/CoreBT/experiments/corebt_linear_probe_v2"
OUTPUT_CSV = os.path.join(ROOT_DIR, "aggregated_results.csv")

rows = []


# Walk experiment folders
for root, dirs, files in os.walk(ROOT_DIR):

    if "summary.json" not in files:
        continue

    summary_path = os.path.join(root, "summary.json")
    experiment_name = os.path.basename(root)

    with open(summary_path, "r") as f:
        data = json.load(f)

    
    # MRI-only training

    if "mri_only" in data:
        for run in data["mri_only"]:

            gm = run["global_metrics"]

            rows.append({
                "experiment": experiment_name,
                "model": "mri_only_train",
                "num_samples": run["summary"]["num_samples"],
                "accuracy": gm["accuracy"],
                "balanced_accuracy": gm["balanced_accuracy"],
                "f1_macro": gm["f1_macro"],
                "f1_weighted": gm["f1_weighted"],
                "precision_macro": gm["precision_macro"],
                "recall_macro": gm["recall_macro"],
                "auroc_macro": gm["auroc_macro"],
                "auprc_macro": gm["auprc_macro"],
                "val_f1": run.get("val_f1"),
            })

    
    # Pathology-only training
    if "histo_only" in data:
        for run in data["histo_only"]:

            gm = run["global_metrics"]

            rows.append({
                "experiment": experiment_name,
                "model": "pathology_only_train",
                "num_samples": run["summary"]["num_samples"],
                "accuracy": gm["accuracy"],
                "balanced_accuracy": gm["balanced_accuracy"],
                "f1_macro": gm["f1_macro"],
                "f1_weighted": gm["f1_weighted"],
                "precision_macro": gm["precision_macro"],
                "recall_macro": gm["recall_macro"],
                "auroc_macro": gm["auroc_macro"],
                "auprc_macro": gm["auprc_macro"],
                "val_f1": run.get("val_f1"),
            })

    
    # Fusion training + ablations

    if "fusion_mri_histo" in data:

        for run in data["fusion_mri_histo"]:
            # print(run)
            test_results = run.get("test_results", {})

            for key, res in test_results.items():

                gm = res["global_metrics"]

                # Map keys to readable names
                if key == "fusion_full":
                    model_name = "fusion_train_full"
                elif key == "mri_only":
                    model_name = "fusion_train_pathology_ablated"
                elif key == "histo_only":
                    model_name = "fusion_train_mri_ablated"
                elif key == "both_ablated":
                    model_name = "fusion_train_both_ablated"
                else:
                    model_name = key

                rows.append({
                    "experiment": experiment_name,
                    "model": model_name,
                    "num_samples": res["summary"]["num_samples"],
                    "accuracy": gm["accuracy"],
                    "balanced_accuracy": gm["balanced_accuracy"],
                    "f1_macro": gm["f1_macro"],
                    "f1_weighted": gm["f1_weighted"],
                    "precision_macro": gm["precision_macro"],
                    "recall_macro": gm["recall_macro"],
                    "auroc_macro": gm["auroc_macro"],
                    "auprc_macro": gm["auprc_macro"],
                    "val_f1": run.get("val_f1"),
                })


# Build DataFrame


df = pd.DataFrame(rows)

# Order models nicely
model_order = [
    "mri_only_train",
    "pathology_only_train",
    "fusion_train_full",
    "fusion_train_mri_ablated",
    "fusion_train_pathology_ablated",
    "fusion_train_both_ablated",
]

df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)

df = df.sort_values(["experiment", "model"])

# Pretty print
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

print("\n============================================================")
print("Aggregated Linear Probe Results")
print("============================================================\n")
print(df.round(4))

# Save CSV
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved aggregated CSV to:\n{OUTPUT_CSV}")



import numpy as np


metrics = [
    ("balanced_accuracy", "Accuracy (Macro)"),
    ("precision_macro", "Precision (Macro)"),
    ("recall_macro", "Recall (Macro)"),
    ("f1_macro", "F1 (Macro)"),
]

grouped = (
    df.groupby(["experiment", "model"])
      .agg(["mean", "std"])
)

# Flatten multiindex columns
grouped.columns = ["_".join(col) for col in grouped.columns]
grouped = grouped.reset_index()


# Generate LaTeX
for experiment in grouped["experiment"].unique():

    print("\n" + r"%% " + "="*70)
    print(f"%% LaTeX Table for {experiment}")
    print(r"%% "+ "="*70 + "\n")

    sub = grouped[grouped["experiment"] == experiment].copy()

    # Determine best per metric (mean only)
    best_per_metric = {}
    for metric, _ in metrics:
        best_per_metric[metric] = sub[f"{metric}_mean"].max()

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(fr"\caption{{Modality ablation study evaluating {' '.join(experiment.split('_'))} classification performance under different modality availability conditions.}}")
    print(r"\label{tab:" + experiment.lower() + r"_ablation}")
    print(r"\renewcommand{\arraystretch}{1.2}")
    print(r"\resizebox{\linewidth}{!}{")
    # print(r"\begin{tabular}{|c|c|c|c|c|c|}")
    print(r"\begin{tabular}{|c|c|c|c|c|}")
    print(r"\hline")
    print("Model Configuration & " +
          " & ".join([m[1] for m in metrics]) +
          r" \\")
    print(r"\hline")

    for _, row in sub.iterrows():

        model_name = row["model"]

        # Pretty names
        name_map = {
            "mri_only_train": "MRI only",
            "pathology_only_train": "Pathology only",
            "fusion_train_full": "MRI + Pathology",
            "fusion_train_mri_ablated": "Fusion (MRI ablated)",
            "fusion_train_pathology_ablated": "Fusion (Pathology ablated)",
            "fusion_train_both_ablated": "Fusion (Both ablated)",
        }

        pretty_name = name_map.get(model_name, model_name)

        line = pretty_name

        for metric, _ in metrics:

            mean_val = row[f"{metric}_mean"]
            std_val = row[f"{metric}_std"]

            if np.isnan(mean_val):
                cell = "--"
            else:
                mean_str = f"{mean_val:.3f}"
                std_str = f"{(0 if np.isnan(std_val) else std_val):.3f}"

                # Bold best
                if mean_val == best_per_metric[metric]:
                    cell = r"\textbf{" + mean_str + "} " 
                else:
                    cell = mean_str

            line += " & " + cell

        line += r" \\"
        print(line)
        print(r"\hline")

    print(r"\end{tabular}")
    print("}")
    print(r"\end{table}")