"""Make plots comparing multi sample performance."""

import functools
import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon
from statannotations.Annotator import Annotator

from cryovit.config import Sample


matplotlib.use("Agg")
colors = sns.color_palette("deep")[:2]
sns.set_theme(style="darkgrid", font="Open Sans")

hue_palette = {
    "3D U-Net": colors[0],
    "CryoViT": colors[1],
}


def read_samples(results_dir: Path, test_samples: str) -> pd.DataFrame:
    """Read sample data from a results directory into a DataFrame.

    Args:
        results_dir (Path): The directory containing sample result file.
        test_samples (str): The test samples represented by a concatenated string.

    Returns:
        pd.DataFrame: A DataFrame containing sample data.
    """
    if not results_dir.exists():
        raise ValueError(f"The directory {results_dir} does not exist")

    return pd.read_csv(results_dir / f"{test_samples}.csv")


def merge_experiments(
    exp_dir: Path,
    exp_names: Dict[str, str],
    samples: str,
    test_samples: str,
) -> pd.DataFrame:
    """Merge experiments data from multiple directories into a single DataFrame.

    Args:
        exp_dir (Path): The directory containing experiment subdirectories.
        exp_names (Dict[str, str]): A dictionary mapping experiment subdirectory names to labels.
        samples (str): The train samples represented by a concatenated string.
        test_samples (str): The test samples represented by a concatenated string.

    Returns:
        pd.DataFrame: A DataFrame containing merged experiment data.
    """
    results = []

    for exp_name, value in exp_names.items():
        results_dir = exp_dir / exp_name / samples / "results"
        df = read_samples(results_dir, test_samples)
        df["Sample"] = df["sample"].apply(lambda x: Sample[x].value)
        df["Model"] = value
        results.append(df.drop(columns=["sample", "TEST_DiceLoss"]))

    return pd.concat(results, axis=0)


def wilcoxon_test(group, model_A: str, model_B: str) -> float:
    """Perform a Wilcoxon signed-rank test between two models on a grouped DataFrame.

    Args:
        group (pd.DataFrame): Grouped DataFrame to perform the test on.
        model_A (str): The name of the first model.
        model_B (str): The name of the second model.

    Returns:
        float: The p-value from the Wilcoxon test.
    """
    score_A = group[group["Model"] == model_A].sort_values("tomo_name").TEST_DiceMetric
    score_B = group[group["Model"] == model_B].sort_values("tomo_name").TEST_DiceMetric

    _, pvalue = wilcoxon(score_A, score_B, method="exact", alternative="greater")
    return pvalue


def plot_df(df: pd.DataFrame, pvalues: pd.Series, title: str, ax: Axes):
    """Plot DataFrame results with box and strip plots including annotations for statistical tests.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        pvalues (pd.Series): Series containing p-values for annotations.
        title (str): The title of the plot.
        file_name (str): Base file name for saving the plot images.
    """
    sample_counts = df["Sample"].value_counts()
    sorted_samples = sample_counts.sort_values(ascending=True).index.tolist()

    params = dict(
        x="Sample",
        y="TEST_DiceMetric",
        hue="Model",
        data=df,
        order=sorted_samples,
    )

    sns.boxplot(
        showfliers=False,
        palette=hue_palette,
        linewidth=1,
        width=0.6,
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

    k1, k2 = df["Model"].unique()
    pairs = [[(s, k1), (s, k2)] for s in pvalues.index]

    annotator = Annotator(ax, pairs, **params)
    annotator.configure(color="blue", line_width=1, verbose=False)
    annotator.set_pvalues_and_annotate(pvalues.values)

    current_labels = ax.get_xticklabels()
    new_labels = [
        f"{label.get_text()}\n(n={sample_counts[label.get_text()] // 2})"
        for label in current_labels
    ]

    ax.set_ylim(-0.05, 1.15)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Dice Score")
    ax.set_xticklabels(new_labels, ha="center")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="lower right", shadow=True)


def compute_stats(df: pd.DataFrame, file_name: str) -> pd.Series:
    """Compute statistical summaries for the DataFrame and save them to a file.

    Args:
        df (pd.DataFrame): The DataFrame to compute statistics on.
        file_name (str): The file path to save the statistics.

    Returns:
        pd.Series: A Series containing p-values for statistical tests.
    """
    grouped = df.groupby(["Sample", "Model"], sort=False)["TEST_DiceMetric"].agg(
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

    pvalues = df.groupby("Sample").apply(test_func)
    pvalues_formatted = pvalues.apply(lambda x: f"{x:.2e}")
    stats_df["p-value"] = pvalues_formatted[stats_df.index]

    sample_counts = df["Sample"].value_counts(ascending=True)
    stats_df = stats_df.loc[sample_counts.index]
    stats_df.reset_index(names="Sample").to_csv(file_name, index=False)

    return pvalues


if __name__ == "__main__":
    exp_dir = Path(sys.argv[1])
    result_dir = exp_dir / "results"
    os.makedirs(result_dir, exist_ok=True)

    exp_names = {
        "multi_sample_cryovit_mito": "CryoViT",
        "multi_sample_unet3d_mito": "3D U-Net",
    }

    healthy_samples = "Q18_Q20_WT"
    diseased_samples = "BACHD_Q109_Q53_Q66_dN17_BACHD"

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[5, 3])  # Adjust the width ratios as needed

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    test_func = functools.partial(wilcoxon_test, model_A="CryoViT", model_B="3D U-Net")

    ################################################

    df = merge_experiments(exp_dir, exp_names, healthy_samples, diseased_samples)
    df.to_csv(result_dir / "multi_sample_HtoD_raw.csv", index=False)
    pvalues = compute_stats(df, result_dir / "multi_sample_HtoD.csv")

    title = "Healthy to Diseased Shift"
    plot_df(df, pvalues, title, ax1)

    ################################################

    df = merge_experiments(exp_dir, exp_names, diseased_samples, healthy_samples)
    df.to_csv(result_dir / "multi_sample_DtoH_raw.csv", index=False)
    pvalues = compute_stats(df, result_dir / "multi_sample_DtoH.csv")

    title = "Diseased to Healthy Shift"
    plot_df(df, pvalues, title, ax2)

    ################################################

    fig.suptitle("CryoViT vs 3D U-Net Comparison Across Domain Shifts")
    fig.supxlabel("Sample Name (Count)")
    # fig.supylabel("Dice Score")

    plt.tight_layout()
    plt.savefig(result_dir / "healthy_diseased_domain_shift.svg")
    plt.savefig(result_dir / "healthy_diseased_domain_shift.png", dpi=300)
