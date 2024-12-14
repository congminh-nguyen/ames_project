import math
from itertools import combinations
from typing import List, Optional, Tuple, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output, display
from scipy import stats
from scipy.stats import gaussian_kde


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


def plot_price_vs_sale_condition(
    df: pd.DataFrame,
    sale_conditions: Optional[Union[str, List[str]]] = None,
    plot_all_data: bool = True,
) -> None:
    """
    Plot scatter plots of sale price vs gross living area, with options to filter by sale conditions.

    Parameters:
    -----------
    df : pandas DataFrame
        The housing data.
    sale_conditions : str or list, optional
        Specific sale conditions to plot. If None, no condition filtering is applied.
    plot_all_data : bool, optional
        If True, plots all data without condition filtering.
    """
    # Prepare data to plot
    plots_to_make = []
    if plot_all_data:
        plots_to_make.append(("All Sales", df))

    if sale_conditions:
        if isinstance(sale_conditions, str):
            sale_conditions = [sale_conditions]
        plots_to_make.extend(
            [
                (f"{cond} Sales", df[df["sale_condition"] == cond])
                for cond in sale_conditions
            ]
        )

    # Set up plot grid
    n_plots = len(plots_to_make)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols  # Correct calculation for number of rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 7 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Create plots
    for idx, (title, data) in enumerate(plots_to_make):
        ax = axes[idx]
        sns.regplot(
            x="gr_liv_area",
            y="saleprice",
            data=data,
            scatter_kws={"alpha": 0.4, "color": "blue"},
            line_kws={"color": "red"},
            ax=ax,
        )

        ax.set_title(title, fontsize=12, pad=15)
        ax.set_xlabel("Gross Living Area (sq ft)", fontsize=10)
        ax.set_ylabel("Sale Price ($)", fontsize=10)

        # Add statistics
        corr = data["gr_liv_area"].corr(data["saleprice"]).round(3)
        ax.text(
            0.05,
            0.95,
            f"n = {len(data)}\nCorrelation = {corr}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Remove unused subplots
    for ax in axes[n_plots:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


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


def interactive_feature_distribution_plots(
    df: pd.DataFrame,
    nominal_list: List[str],
    ordinal_list: List[str],
    continuous_list: List[str],
    discrete_list: List[str],
) -> None:
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


def interactive_feature_target_distribution_plots(
    df: pd.DataFrame,
    nominal_list: List[str],
    ordinal_list: List[str],
    continuous_list: List[str],
    discrete_list: List[str],
    target_col: str,
) -> None:
    """
    Interactive function to plot feature relationships with target variable based on feature type.
    For categorical features (nominal/ordinal): Shows box plots of target by category
    For numerical features (continuous/discrete): Shows scatter plots with trend lines

    Args:
        df (pd.DataFrame): DataFrame containing the data
        nominal_list (list): List of nominal features
        ordinal_list (list): List of ordinal features
        continuous_list (list): List of continuous features
        discrete_list (list): List of discrete features
        target_col (str): Name of target variable column. Defaults to 'saleprice'
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

    def plot_feature_target_relationships(*args):
        """Plot relationships between features and target based on selected type"""
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
                    # Calculate mean target value for each category and sort
                    category_means = (
                        df.groupby(feature)[target_col].mean().sort_values()
                    )
                    category_order = category_means.index.tolist()

                    # Box plot for categorical variables vs target with ordered categories
                    sns.boxplot(
                        x=df[feature], y=df[target_col], ax=ax, order=category_order
                    )

                    # Rotate x-axis labels properly
                    ticks = ax.get_xticks()
                    labels = [label.get_text() for label in ax.get_xticklabels()]
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(labels, rotation=45, ha="right")

                    # Add category counts
                    counts = df[feature].value_counts()
                    stats_text = f"n categories: {len(counts)}\n"
                    stats_text += f"Most common: {counts.index[0]} ({counts.iloc[0]})"

                else:  # Continuous or Discrete
                    # Scatter plot with trend line
                    sns.scatterplot(data=df, x=feature, y=target_col, alpha=0.5, ax=ax)
                    sns.regplot(
                        data=df,
                        x=feature,
                        y=target_col,
                        scatter=False,
                        color="red",
                        ax=ax,
                    )

                    # Add correlation coefficient
                    corr = df[feature].corr(df[target_col])
                    stats_text = f"Correlation: {corr:.3f}\n"
                    stats_text += f"Mean: {df[feature].mean():.2f}\n"
                    stats_text += f"Std: {df[feature].std():.2f}"

                ax.set_title(f"{feature} vs {target_col}")
                ax.set_xlabel(feature)
                ax.set_ylabel(target_col)

                # Add stats text box
                ax.text(
                    0.95,
                    0.95,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()

    # Set up observer
    type_select.observe(plot_feature_target_relationships, "value")

    # Display widgets
    display(widgets.VBox([type_select, output]))


def anova_categorical_feature_importance(
    df: pd.DataFrame,
    cat_features: List[str],
    target_col: str = "saleprice",
    top_n: int = 10,
) -> None:
    """
    Performs one-way ANOVA and calculates effect sizes for categorical features.

    Args:
        df (pd.DataFrame): Input dataframe
        cat_features (list): List of categorical feature names
        target_col (str): Name of target variable column
        top_n (int): Number of top features to display by effect size

    Returns:
        Displays bar plot of top features by effect size
    """
    # Perform one-way ANOVA and calculate effect sizes
    anova_results = []

    for cat in cat_features:
        # Get groups for this categorical feature
        groups = [group[target_col].values for name, group in df.groupby(cat)]

        # Calculate ANOVA
        f_val, p_val = stats.f_oneway(*groups)

        # Calculate effect size (eta squared)
        # First calculate sum of squares between groups
        group_means = [np.mean(g) for g in groups]
        grand_mean = np.mean([x for g in groups for x in g])
        n_groups = [len(g) for g in groups]
        ss_between = sum(
            n * (xbar - grand_mean) ** 2 for n, xbar in zip(n_groups, group_means)
        )

        # Calculate total sum of squares
        ss_total = sum((x - grand_mean) ** 2 for g in groups for x in g)

        # Calculate eta squared
        eta_sq = ss_between / ss_total

        # Get number of categories and total observations
        n_categories = len(groups)
        n_total = sum(n_groups)

        anova_results.append(
            {
                "Feature": cat,
                "F-Value": f_val,
                "P-Value": p_val,
                "Effect Size (η²)": eta_sq,
                "Categories": n_categories,
                "Total Obs": n_total,
            }
        )

    # Create and sort DataFrame
    anova_df = pd.DataFrame(anova_results)
    anova_df.sort_values(
        by=["P-Value", "Effect Size (η²)"], ascending=[True, False], inplace=True
    )

    # Add significance indicators
    anova_df["Significance"] = pd.cut(
        anova_df["P-Value"],
        bins=[-np.inf, 0.001, 0.01, 0.05, np.inf],
        labels=["***", "**", "*", "ns"],
    )

    # Format P-values scientifically
    anova_df["P-Value"] = anova_df["P-Value"].apply(lambda x: f"{x:.2e}")
    anova_df["Effect Size (η²)"] = anova_df["Effect Size (η²)"].apply(
        lambda x: f"{x:.3f}"
    )

    plt.figure(figsize=(12, 6))
    top_features = anova_df.head(top_n).copy()
    top_features["Effect Size (η²)"] = top_features["Effect Size (η²)"].astype(float)

    sns.barplot(
        data=top_features,
        x="Effect Size (η²)",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
    )

    plt.title(f"Top {top_n} Categorical Features by Effect Size (η²)")
    plt.xlabel("Effect Size (η²)")
    plt.tight_layout()
    plt.show()


def plot_cramer_v_associations(
    df: pd.DataFrame, cat_features: List[str], top_n: int = 50
) -> None:
    """
    Calculate and plot Cramer's V associations between categorical features.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        cat_features (list): List of categorical feature names to analyze
        top_n (int): Number of top feature pairs to plot. Defaults to 50.
    """
    try:
        # Generate feature pairs
        cat_feat_combos = combinations(cat_features, 2)
        chisq_df_rows = []

        # Calculate Cramer's V for each pair
        N = df.shape[0]
        for cat1, cat2 in cat_feat_combos:
            cat_crosstab = pd.crosstab(df[cat1], df[cat2], normalize=False)
            chisq, p, dof, _ = stats.chi2_contingency(cat_crosstab)

            # Calculate Cramer's V
            k = min(len(df[cat1].unique()), len(df[cat2].unique()))
            cramersV = math.sqrt(chisq / (N * (k - 1)))

            chisq_df_rows.append((cat1, cat2, chisq, p, cramersV, k))

        # Create and sort dataframe
        chisq_df = pd.DataFrame(
            chisq_df_rows, columns=["cat1", "cat2", "chisq", "p", "Cramer's V", "dof"]
        )
        chisq_df.sort_values(by="Cramer's V", ascending=True, inplace=True)
        chisq_df.reset_index(drop=True, inplace=True)

        # Create figure for plotting
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 10))

        # Plot top feature pairs
        top_pairs = chisq_df.tail(top_n)
        ax1.barh(
            y=range(len(top_pairs)),
            width=top_pairs["Cramer's V"],
            tick_label=[
                f"{cat1} - {cat2}"
                for cat1, cat2 in zip(top_pairs["cat1"], top_pairs["cat2"])
            ],
        )
        ax1.set_title(f"Top {top_n} Categorical Feature Pairs by Cramer's V")
        ax1.set_xlabel("Cramer's V")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")


def plot_numerical_correlation_matrix(
    df: pd.DataFrame,
    numerical_cols: List[str],
    target: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 18),
    cmap_start: int = 230,
    cmap_end: int = 20,
) -> None:
    """
    Plot a correlation matrix for numerical features.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        numerical_cols (List[str]): List of numerical column names to include in the correlation matrix.
        target (Optional[str]): Name of the target variable. If provided, the features will be sorted by correlation with this target.
        figsize (Tuple[int, int]): Size of the figure. Defaults to (20, 18).
        cmap_start (int): Start color for the diverging palette. Defaults to 230.
        cmap_end (int): End color for the diverging palette. Defaults to 20.
    """
    # Include target in numerical columns if provided
    if target and target not in numerical_cols:
        numerical_cols = [target] + numerical_cols

    # Calculate correlation matrix for numerical features
    corr = df[numerical_cols].corr()

    # Sort features by correlation with target if target is provided
    if target:
        target_corr = corr[target].sort_values(ascending=False)
        sorted_features = [target] + list(target_corr.index.drop(target))
        corr = corr.loc[sorted_features, sorted_features]

    # Create figure and axis with specified size
    fig, ax = plt.subplots(figsize=figsize)

    # Create custom diverging colormap
    cmap = sns.diverging_palette(cmap_start, cmap_end, as_cmap=True)

    # Create heatmap with improved visibility
    sns.heatmap(
        corr,
        xticklabels=corr.columns.values,
        yticklabels=corr.index.values,
        cmap=cmap,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10},
    )

    # Customize axis labels with larger fonts
    title = "Correlation Matrix of Numerical Features"
    if target:
        title += f" (Including {target.capitalize()})"
    plt.title(title, pad=20, size=16)
    ax.xaxis.tick_top()
    plt.xticks(rotation=45, ha="left", size=12)
    plt.yticks(rotation=0, size=12)

    # Add more padding to prevent label cutoff
    plt.tight_layout(pad=1.5)
    plt.show()
