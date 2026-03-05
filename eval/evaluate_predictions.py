import argparse
import json, os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    confusion_matrix,
)

TASK_MAP = {
    "level1": ("level1_label", "level1_pred"),
    "lgghgg": ("lgghgg_label", "lgghgg_pred"),
    "who": ("who_grade_label", "who_grade_pred"),
}


def evaluate(df, true_col, pred_col):

    y_true = df[true_col].values
    y_pred = df[pred_col].values

    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))

    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    balanced_acc = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    labels = np.unique(y_true)

    precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {}

    total_samples = len(y_true)

    for i, cls in enumerate(labels):
        per_class[int(cls)] = {
            "support": int(support_c[i]),
            "fraction": float(support_c[i] / total_samples),
            "precision": float(precision_c[i]),
            "recall": float(recall_c[i]),
            "f1": float(f1_c[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    results = {
        "summary": {
            "num_samples": int(total_samples)
        },
        "global_metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
        },
        "per_class_metrics": per_class,
        "confusion_matrix": {
            "labels": labels.tolist(),
            "matrix": cm.tolist(),
        },
    }

    return results


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--submission_csv", required=True)
    parser.add_argument("--reference_csv", required=False, default='corebt_sharedtest_groundtruth_alltasks_trainval.csv')
    parser.add_argument(
        "--task",
        required=True,
        nargs="+",
        choices=["level1", "lgghgg", "who", "all"]
    )
    parser.add_argument("--run_id", default="run")
    parser.add_argument("--output_json", default=None)

    args = parser.parse_args()

    tasks = list(TASK_MAP.keys()) if "all" in args.task else args.task

    ref = pd.read_csv(args.reference_csv)
    sub = pd.read_csv(args.submission_csv)

    if "subject_id" not in sub.columns:
        raise ValueError("Submission must contain 'subject_id' column")

    if len(sub.subject_id.unique()) != len(sub):
        raise ValueError("Duplicate subject_ids detected in submission")

    df = ref.merge(sub, on="subject_id", how="inner")

    if len(df) != len(ref):
        missing = set(ref.subject_id) - set(sub.subject_id)
        raise ValueError(f"Missing predictions for subjects: {missing}")

    if args.output_json is not None:

        if os.path.exists(args.output_json):
            with open(args.output_json, "r") as f:
                all_results = json.load(f)
        else:
            all_results = {"runs": {}}

        if args.run_id not in all_results["runs"]:
            all_results["runs"][args.run_id] = {"tasks": {}}

    for task in tasks:

        true_col, pred_col = TASK_MAP[task]

        if pred_col not in sub.columns:
            raise ValueError(f"Submission must contain '{pred_col}' column")

        results = evaluate(df, true_col, pred_col)

        results["task"] = task
        results["run_id"] = args.run_id

        leaderboard_scores = {
            "accuracy": results["global_metrics"]["accuracy"],
            "balanced_accuracy": results["global_metrics"]["balanced_accuracy"],
            "f1_macro": results["global_metrics"]["f1_macro"],
        }

        if args.output_json is not None:
            all_results["runs"][args.run_id]["tasks"][task] = results

        print(f"\n{task}")
        print(json.dumps(leaderboard_scores, indent=2))

    if args.output_json is not None:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()