from typing import Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output, display
from scipy import stats
from scipy.stats import gaussian_kde


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
    )
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
        df (pd.DataFrame): DataFrame containing the data
        x (str): The column name to be plotted on the x-axis
        y (str): The column name to be plotted on the y-axis
        hue (str): The column name to be used for color encoding
        num_subplots (int, optional): Number of subplots to create. Defaults to 4
        figsize (Tuple[int, int], optional): Size of the figure. Defaults to (16, 20)
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


def output_distribution_plotting(
    df: pd.DataFrame, target_col: str, alpha: float = 0.10
) -> None:
    """
    Analyze and plot the distribution of a target variable.

    Args:
        df (pd.DataFrame): DataFrame containing the target variable
        target_col (str): Name of target variable column
        alpha (float, optional): Significance level for statistical tests. Defaults to 0.10
    """
    # Basic statistics
    # Price range distribution
    print("\nPrice Range Distribution (%):")
    price_ranges = pd.cut(
        df[target_col],
        bins=[0, 100000, 200000, 300000, 400000, float("inf")],
        labels=["<$100k", "$100k-$200k", "$200k-$300k", "$300k-$400k", ">$400k"],
    )
    print(
        price_ranges.value_counts(normalize=True)
        .mul(100)
        .round(1)
        .sort_index()
        .to_string()
    )

    # Normality tests
    print(f"\nNormality Tests (α = {alpha}):")
    k2, p1 = stats.normaltest(df[target_col])
    stat, p2 = stats.shapiro(df[target_col])
    result = stats.anderson(df[target_col])
    jb_stat, p3 = stats.jarque_bera(df[target_col])

    normality_tests = pd.DataFrame(
        {
            "Test": [
                "D'Agostino-Pearson",
                "Shapiro-Wilk",
                "Anderson-Darling",
                "Jarque-Bera",
            ],
            "Test Statistic": [k2, stat, result.statistic, jb_stat],
            "p-value": [p1, p2, None, p3],
            "Result": [
                "Reject H₀" if p1 < alpha else "Fail to reject H₀",
                "Reject H₀" if p2 < alpha else "Fail to reject H₀",
                (
                    "Reject H₀"
                    if result.statistic > result.critical_values[0]
                    else "Fail to reject H₀"
                ),
                "Reject H₀" if p3 < alpha else "Fail to reject H₀",
            ],
        }
    )
    print(normality_tests.to_string(index=False))

    # Fat-tailed tests
    print(f"\nFat-Tailed Tests (α = {alpha}):")
    kurtosis = stats.kurtosis(df[target_col], fisher=False)
    tail_tests = pd.DataFrame(
        {
            "Test": ["Kurtosis"],
            "Test Statistic": [kurtosis],
            "Reference Value": [3.0],  # Normal distribution kurtosis
            "Result": ["Fat-tailed" if kurtosis > 3.0 else "Not fat-tailed"],
        }
    )
    print(tail_tests.to_string(index=False))

    # Distribution plots
    plt.figure(figsize=(14, 4))
    plt.hist(df[target_col], bins=50, rwidth=0.8, alpha=0.7)
    density = gaussian_kde(df[target_col])
    xs = np.linspace(df[target_col].min(), df[target_col].max(), 200)
    density_scaled = (
        density(xs)
        * len(df[target_col])
        * (df[target_col].max() - df[target_col].min())
        / 50
    )
    plt.plot(xs, density_scaled, linewidth=2, label="Density")
    plt.title("House Price Distribution in Ames, Iowa")
    plt.xlabel("Sale Price ($)")
    plt.ylabel("Number of Houses")
    plt.legend()
    plt.grid(False)
    plt.show()

    # QQ plot
    plt.figure(figsize=(14, 6))
    stats.probplot(df[target_col], dist="norm", plot=plt)
    plt.title("QQ Plot of Sale Price")
    plt.show()


def plot_target_over_time(
    df: pd.DataFrame, target_col: str, year_col: str, month_col: str
) -> None:
    """
    Plot number of houses sold and average sale price over time.

    Args:
        df (pd.DataFrame): DataFrame containing the sales data
        target_col (str): Name of column containing sale prices
        year_col (str): Name of column containing sale year
        month_col (str): Name of column containing sale month
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot the count of houses sold
    color = "tab:blue"
    ax1.set_xlabel("Year-Month")
    ax1.set_ylabel("Number of Houses Sold", color=color)
    house_count = df.groupby([year_col, month_col]).pid.count()
    house_count.plot(kind="bar", ax=ax1, color=color, alpha=0.6, label="Houses Sold")
    ax1.tick_params(axis="y", labelcolor=color)

    # Create a second y-axis for the average sale price
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Average Sale Price ($)", color=color)
    avg_price = df.groupby([year_col, month_col])[target_col].mean()
    avg_price.plot(
        ax=ax2,
        color=color,
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Avg Sale Price",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    # Add titles and legends
    plt.title("Houses Sold and Average Sale Price by Month-Year")
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # Remove grid
    ax1.grid(False)
    ax2.grid(False)

    plt.show()


def interactive_feature_distribution_plots(
    df: pd.DataFrame,
    nominal_list: list,
    ordinal_list: list,
    continuous_list: list,
    discrete_list: list,
):
    """
    Interactive function to plot distributions for features based on their type.
    Allows user to select feature type and plots distributions for all relevant features.
    Uses provided feature type lists: nominal, ordinal, continuous, discrete.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        nominal_list (list): List of nominal features.
        ordinal_list (list): List of ordinal features.
        continuous_list (list): List of continuous features.
        discrete_list (list): List of discrete features.
    """
    # Create feature type dropdown
    type_select = widgets.Dropdown(
        options=["Nominal", "Ordinal", "Continuous", "Discrete"],
        description="Feature Type:",
        style={"description_width": "initial"},
        layout={"width": "300px"},
    )

    # Create output widget for plots
    output = widgets.Output()

    def plot_individual_feature_distributions(*args):
        """Plot distributions for all features of the selected type"""
        with output:
            clear_output(wait=True)
            type_map = {
                "Nominal": nominal_list,
                "Ordinal": ordinal_list,
                "Continuous": continuous_list,
                "Discrete": discrete_list,
            }
            selected_features = type_map[type_select.value]

            num_plots = len(selected_features)
            num_cols = 4
            num_rows = (num_plots + num_cols - 1) // num_cols

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
            axes = axes.flatten()

            for i, feature in enumerate(selected_features):
                ax = axes[i]

                if type_select.value in ["Nominal", "Ordinal"]:
                    # Bar plot for categorical variables
                    value_counts = (
                        df[feature].value_counts().sort_values(ascending=False)
                    )
                    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45, ha="right")

                elif type_select.value == "Discrete":
                    # Bar plot with count for discrete numerical variables
                    sns.histplot(data=df, x=feature, discrete=True, ax=ax)

                else:  # Continuous
                    # Histogram with KDE for continuous variables
                    sns.histplot(data=df, x=feature, kde=True, ax=ax)

                ax.set_title(f"{feature} ({type_select.value})")
                ax.set_xlabel(feature)
                ax.set_ylabel("Count")

                # Add summary statistics
                if type_select.value in ["Discrete", "Continuous"]:
                    stats_text = f"Mean: {df[feature].mean():.2f}\n"
                    stats_text += f"Median: {df[feature].median():.2f}\n"
                    stats_text += f"Std: {df[feature].std():.2f}"
                else:
                    stats_text = f"Mode: {df[feature].mode()[0]}\n"
                    stats_text += f"Unique Values: {df[feature].nunique()}"

                ax.text(
                    0.95,
                    0.95,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()

    # Set up observer
    type_select.observe(plot_individual_feature_distributions, "value")

    # Display widgets
    display(widgets.VBox([type_select, output]))
