from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_missing_data_heatmap(df: pd.DataFrame) -> None:
    """
    Plot a heatmap of missing data, with columns sorted by missing percentage.

    Args:
        df (pd.DataFrame): Input DataFrame to analyze and plot
    """
    # Identify columns with missing values
    missing_columns = df.columns[df.isnull().any()].tolist()

    # Sort columns by missing percentage
    missing_columns = sorted(
        missing_columns, key=lambda col: df[col].isnull().mean(), reverse=True
    )

    # Visualize missing data by a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[missing_columns].isnull(), cbar=False, cmap="viridis")
    plt.title("Heatmap of Missing Data (Sorted by Missing Percentage)")
    plt.xticks(
        ticks=np.arange(len(missing_columns)) + 0.5,
        labels=missing_columns,
        rotation=90,
        ha="center",
    )  # Adjust x-axis labels
    plt.show()


def box_plot_dist(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    num_subplots: int = 4,
    figsize: Tuple[int, int] = (16, 20),
) -> None:
    """
    Plots the distribution of a specified x variable by y and hue variables using subplots.

    Parameters:
    - df: DataFrame containing the data.
    - x: The column name to be plotted on the x-axis.
    - y: The column name to be plotted on the y-axis.
    - hue: The column name to be used for color encoding.
    - num_subplots: Number of subplots to create.
    - figsize: Size of the figure.

    Returns:
    - None
    """
    unique_y_values = df[y].unique()
    y_chunks = np.array_split(unique_y_values, num_subplots)

    fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=figsize, sharex=True)

    for i, ax in enumerate(axes):
        sns.boxplot(
            data=df[df[y].isin(y_chunks[i])],
            x=x,
            y=y,
            hue=hue,
            orient="h",
            palette="Set2",
            ax=ax,
        )
        ax.set_title(
            f"Distribution of {x} by {y} and {hue} (Part {i + 1})", fontsize=14
        )
        ax.set_ylabel(y, fontsize=12)
        ax.set_xlabel(x, fontsize=12)
        ax.legend(title=hue, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()
