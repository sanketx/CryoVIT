"""Make plots comparing fractional LOO sample performance."""

import functools
import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from statannotations.Annotator import Annotator

from cryovit.config import Sample


matplotlib.use("Agg")
colors = sns.color_palette("deep")[:3]
sns.set_theme(style="darkgrid", font="Open Sans")

hue_palette = {
    "3D U-Net": colors[0],
    "CryoViT": colors[1],
    "CryoViT with Sparse Labels": colors[1],
    "CryoViT with Dense Labels": colors[2],
}


def merge_samples(results_dir: Path) -> pd.DataFrame:
    """Merge sample data from a results directory into a single DataFrame.

    Args:
        results_dir (Path): The directory containing sample result files.

    Returns:
        pd.DataFrame: A DataFrame containing merged sample data.
    """
    if not results_dir.exists():
        raise ValueError(f"The directory {results_dir} does not exist")

    results = []

    for split in range(1, 11):
        for sample in Sample:
            file_name = results_dir / f"split_{split}" / f"{sample.name}.csv"
            df = pd.read_csv(file_name)
            df["Sample"] = sample.value
            df["split"] = split
            results.append(df.drop(columns=["sample", "TEST_DiceLoss"]))

    return pd.concat(results, axis=0)


def merge_experiments(
    exp_dir: Path, exp_names: Dict[str, str], key: str
) -> pd.DataFrame:
    """Merge experiments data from multiple directories into a single DataFrame.

    Args:
        exp_dir (Path): The directory containing experiment subdirectories.
        exp_names (Dict[str, str]): A dictionary mapping experiment subdirectory names to labels.
        key (str): The column name to assign to the experiment labels in the merged DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing merged experiment data.
    """
    results = []

    for exp_name, value in exp_names.items():
        results_dir = exp_dir / exp_name / "results"
        df = merge_samples(results_dir)
        df[key] = value
        results.append(df)

    return pd.concat(results, axis=0)


def significance_test(group, key: str, model_A: str, model_B: str) -> float:
    """Perform a paired t-test between two models on a grouped DataFrame.

    Args:
        group (pd.DataFrame): Grouped DataFrame to perform the test on.
        key (str): Column name to filter the DataFrame on model names.
        model_A (str): The name of the first model.
        model_B (str): The name of the second model.

    Returns:
        float: The p-value from the paired t-test.
    """
    score_A = group[group[key] == model_A].sort_values("tomo_name").TEST_DiceMetric
    score_B = group[group[key] == model_B].sort_values("tomo_name").TEST_DiceMetric

    _, pvalue = ttest_rel(score_A, score_B, alternative="greater")
    return pvalue


def plot_df(df: pd.DataFrame, pvalues: pd.Series, key: str, title: str, file_name: str):
    """Plot DataFrame results with box and strip plots including annotations for statistical tests.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        pvalues (pd.Series): Series containing p-values for annotations.
        key (str): The column name used to group data points in the plot.
        title (str): The title of the plot.
        file_name (str): Base file name for saving the plot images.
    """
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    params = dict(
        x="split",
        y="TEST_DiceMetric",
        hue=key,
        data=df,
    )

    sns.boxplot(
        showfliers=False,
        palette=hue_palette,
        linewidth=1,
        medianprops=dict(linewidth=2, color="firebrick"),
        ax=ax,
        **params,
    )
    sns.stripplot(
        dodge=True,
        marker=".",
        alpha=0.5,
        palette="dark:black",
        ax=ax,
        **params,
    )

    k1, k2 = df[key].unique()
    pairs = [[(s, k1), (s, k2)] for s in pvalues.index]

    annotator = Annotator(ax, pairs, **params)
    annotator.configure(color="blue", line_width=1, verbose=False)
    annotator.set_pvalues_and_annotate(pvalues.values)

    current_labels = ax.get_xticklabels()
    new_labels = [f"{label.get_text()}0%" for label in current_labels]
    ax.set_xticklabels(new_labels, ha="center")

    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.suptitle(title)
    fig.supxlabel("Fraction of Training Data")
    fig.supylabel("Dice Score")

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], loc="lower center", shadow=True)

    plt.tight_layout()
    plt.savefig(f"{file_name}.svg")
    plt.savefig(f"{file_name}.png", dpi=300)


def compute_stats(df: pd.DataFrame, key: str, file_name: str) -> pd.Series:
    """Compute statistical summaries for the DataFrame and save them to a file.

    Args:
        df (pd.DataFrame): The DataFrame to compute statistics on.
        key (str): The column name used to group data for statistics.
        file_name (str): The file path to save the statistics.

    Returns:
        pd.Series: A Series containing p-values for statistical tests.
    """
    grouped = df.groupby(["split", key], sort=False)["TEST_DiceMetric"].agg(
        mean="mean",
        std="std",
        median="median",
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75),
    )

    transforms = {
        "Median Dice Score": lambda row: f"{row['median']:.2f}",
        "Mean Dice Score ± Std": lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}",
        "Dice Score Quartiles (Q1 - Q3)": lambda row: f"{row['Q1']:.2f} - {row['Q3']:.2f}",
    }

    values = {col: grouped.apply(func, axis=1) for col, func in transforms.items()}
    stats_df = pd.DataFrame.from_dict(values).unstack(level=-1)

    pvalues = df.groupby("split").apply(test_func)
    pvalues_formatted = pvalues.apply(lambda x: f"{x:.2e}")

    stats_df["p-value"] = pvalues_formatted[stats_df.index]
    stats_df.reset_index(names="split").to_csv(file_name, index=False)

    return pvalues


if __name__ == "__main__":
    exp_dir = Path(sys.argv[1])
    result_dir = exp_dir / "results"
    os.makedirs(result_dir, exist_ok=True)

    exp_names = {
        "fractional_sample_cryovit_mito": "CryoViT",
        "fractional_sample_unet3d_mito": "3D U-Net",
    }

    test_func = functools.partial(
        significance_test, key="Model", model_A="CryoViT", model_B="3D U-Net"
    )

    df = merge_experiments(exp_dir, exp_names, key="Model")
    df.to_csv(result_dir / "fractional_sample_model_raw.csv", index=False)
    pvalues = compute_stats(df, "Model", result_dir / "fractional_sample_model.csv")

    title = "CryoViT vs 3D U-Net Comparison on All Samples"
    plot_df(df, pvalues, "Model", title, result_dir / "fractional_sample_model")

    #################################################

    exp_names = {
        "fractional_sample_cryovit_mito": "CryoViT with Sparse Labels",
        "fractional_sample_cryovit_mito_ai": "CryoViT with Dense Labels",
    }

    test_func = functools.partial(
        significance_test,
        key="Label Type",
        model_A="CryoViT with Dense Labels",
        model_B="CryoViT with Sparse Labels",
    )

    df = merge_experiments(exp_dir, exp_names, key="Label Type")
    df.to_csv(result_dir / "fractional_sample_label_raw.csv", index=False)
    pvalues = compute_stats(
        df, "Label Type", result_dir / "fractional_sample_label.csv"
    )

    title = "CryoVIT: Sparse vs Dense Labels comparison on All Samples"
    plot_df(df, pvalues, "Label Type", title, result_dir / "fractional_sample_label")
