"""Splits

Creates a master split file from all the annotation csvs.
Each tomogram is assigned 3 split IDs - LOO, 5-fold, and 10-fold.
"""

import os

import pandas as pd
from sklearn.model_selection import KFold

from .. import config

SEED = 0


def split(df, sample, K):
    annotation_file = os.path.join(config.ANNOTATION_CSV, f"{sample}.csv")

    if not os.path.exists(annotation_file):
        raise RuntimeError(f"No annotations for {sample} were found")

    num_samples = df.shape[0]
    n_splits = num_samples if num_samples < K or K == 0 else K

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    X = [[0] for _ in range(num_samples)]
    split_id = [-1 for _ in range(num_samples)]

    for f, (_, test) in enumerate(kf.split(X)):
        for idx in test:
            split_id[idx] = f

    return split_id


def main():
    dfs = []

    for sample in config.DATASETS:
        annotation_file = os.path.join(config.ANNOTATION_CSV, f"{sample}.csv")
        df = pd.read_csv(annotation_file)

        split_loo = split(df, sample, 0)
        split_5 = split(df, sample, 5)
        split_10 = split(df, sample, 10)

        df["split_loo"] = split_loo
        df["split_5"] = split_5
        df["split_10"] = split_10
        df["sample"] = sample
        dfs.append(df)

    dst_path = os.path.join(config.EXP_DIR, "splits.csv")
    split_df = pd.concat(dfs)
    split_df.to_csv(dst_path, index=False)


if __name__ == "__main__":
    main()
