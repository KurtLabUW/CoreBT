

import os
import pandas as pd

def load_single_dir(base_dir, tasks):
    dfs = []

    for task in tasks:
        csv_path = os.path.join(base_dir, task, 'test_predictions.csv')

        if not os.path.exists(csv_path):
            print(f"Warning: missing {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        pred_col = f"{task}_pred"
        if pred_col not in df.columns:
            raise ValueError(f"{csv_path} missing column {pred_col}")

        dfs.append(df[['subject_id', pred_col]])

    if not dfs:
        return None

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on='subject_id', how='outer')

    return merged


def merge_predictions_with_priority(base_dirs, output_path=None):
    """
    base_dirs: list of directories in PRIORITY order
               first = highest priority
    """
    tasks = ['level1', 'lgghgg', 'who_grade']

    merged_priority = None

    for base_dir in base_dirs:
        df = load_single_dir(base_dir, tasks)
        if df is None:
            continue

        df = df.set_index('subject_id')

        if merged_priority is None:
            merged_priority = df
        else:
            # keep existing values, fill missing from df
            merged_priority = merged_priority.combine_first(df)

    if merged_priority is None:
        raise ValueError("No valid data found in any directory")

    merged_priority = merged_priority.reset_index()
    merged_priority = merged_priority.sort_values('subject_id').reset_index(drop=True)

    if output_path is None:
        output_path = os.path.join(base_dirs[0], 'merged_test_predictions_priority.csv')

    merged_priority.to_csv(output_path, index=False)
    print(f"Saved merged CSV to {output_path}")

    return merged_priority

if __name__ == '__main__':
    base_dirs=['run/mri', 'run/histopathology']
    merge_predictions_with_priority(base_dirs, output_path='run/prediction.csv')
    print('yep')