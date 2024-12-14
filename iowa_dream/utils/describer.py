import pandas as pd


def get_value_info(df: pd.DataFrame, col: str, is_categorical: bool) -> str:
    """
    Get value information for a column in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        col (str): Column name
        is_categorical (bool): Flag indicating if the column is categorical

    Returns:
        str: Value information string
    """
    if is_categorical:
        unique_vals = sorted(df[col].dropna().unique())
        return f"{len(unique_vals)} unique values: {', '.join(map(str, unique_vals))}"
    else:
        unique_count = len(df[col].unique())
        return (
            f"Range: {df[col].min()} to {df[col].max()} ({unique_count} unique values)"
        )


def categorical_describer(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    """
    Generate a comprehensive analysis of categorical columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to analyze
        categorical_columns (list): List of categorical column names

    Returns:
        pd.DataFrame: DataFrame containing detailed categorical column analysis with columns:
            - Total Missing: Count of missing values
            - Percent Missing: Percentage of missing values
            - Data Type: Column data type
            - Unique Values: Count of distinct values
            - Value Information: Full listing of unique values
            - Mode: Most frequent value
            - Mode Frequency: Frequency of the most frequent value
    """
    # Filter categorical columns
    cat_cols = [col for col in categorical_columns if col in df.columns]

    # Analyze missing data and types
    total = df[cat_cols].isnull().sum()
    percent = df[cat_cols].isnull().sum() / df[cat_cols].shape[0]
    dtypes = df[cat_cols].dtypes

    # Get unique value information
    value_info = {col: get_value_info(df, col, True) for col in cat_cols}

    # Get mode and mode frequency
    mode = df[cat_cols].mode().iloc[0]
    mode_freq = df[cat_cols].apply(lambda x: x.value_counts().max())

    # Create comprehensive DataFrame
    column_analysis = pd.concat(
        [total, percent, dtypes, pd.Series(value_info), mode, mode_freq],
        axis=1,
        keys=[
            "Total Missing",
            "Percent Missing",
            "Data Type",
            "Value Information",
            "Mode",
            "Mode Frequency",
        ],
    )

    # Sort by percent missing
    column_analysis = column_analysis.sort_values("Percent Missing", ascending=False)

    # Print summary
    print("\nCategorical Columns Summary:")
    print(f"Total categorical columns: {len(cat_cols)}")
    print(f"Columns with missing values: {(total > 0).sum()}")

    return column_analysis


def numerical_describer(df: pd.DataFrame, numerical_columns: list) -> pd.DataFrame:
    """
    Generate a comprehensive analysis of numerical columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to analyze
        numerical_columns (list): List of numerical column names

    Returns:
        pd.DataFrame: DataFrame containing detailed numerical column analysis with columns:
            - Total Missing: Count of missing values
            - Percent Missing: Percentage of missing values
            - Data Type: Column data type
            - Value Information: Range and uniqueness information
            - Mean: Mean of the column (ignores missing values)
            - Std Dev: Standard deviation of the column (ignores missing values)
            - Min: Minimum value (ignores missing values)
            - 25%: 25th percentile (ignores missing values)
            - 50%: Median value (ignores missing values)
            - 75%: 75th percentile (ignores missing values)
            - Max: Maximum value (ignores missing values)
    """
    # Filter numerical columns
    num_cols = [col for col in numerical_columns if col in df.columns]

    # Analyze missing data and types
    total = df[num_cols].isnull().sum()
    percent = df[num_cols].isnull().sum() / df[num_cols].shape[0]
    dtypes = df[num_cols].dtypes

    # Get value range information
    value_info = {col: get_value_info(df, col, False) for col in num_cols}

    # Get statistical information, ignoring missing values
    mean = df[num_cols].mean()
    std_dev = df[num_cols].std()
    min_val = df[num_cols].min()
    q25 = df[num_cols].quantile(0.25)
    median = df[num_cols].median()
    q75 = df[num_cols].quantile(0.75)
    max_val = df[num_cols].max()

    # Create comprehensive DataFrame
    column_analysis = pd.concat(
        [
            total,
            percent,
            dtypes,
            pd.Series(value_info),
            mean,
            std_dev,
            min_val,
            q25,
            median,
            q75,
            max_val,
        ],
        axis=1,
        keys=[
            "Total Missing",
            "Percent Missing",
            "Data Type",
            "Value Information",
            "Mean",
            "Std Dev",
            "Min",
            "25%",
            "50%",
            "75%",
            "Max",
        ],
    )

    # Sort by percent missing
    column_analysis = column_analysis.sort_values("Percent Missing", ascending=False)

    # Print summary
    print("\nNumerical Columns Summary:")
    print(f"Total numerical columns: {len(num_cols)}")
    print(f"Columns with missing values: {(total > 0).sum()}")

    return column_analysis


def analyze_categorical_sparsity(
    df: pd.DataFrame, threshold_dominant: float = 0.95, threshold_rare: float = 0.01
) -> pd.DataFrame:
    """
    Analyze categorical features for sparsity and imbalanced distributions.

    Identifies features with dominant categories and rare categories based on specified thresholds.
    Calculates distribution statistics for each categorical feature.

    Args:
        df: Input DataFrame to analyze
        threshold_dominant: Threshold for flagging features with a dominant category (default 0.9)
        threshold_rare: Threshold for flagging rare categories (default 0.05)

    Returns:
        DataFrame containing analysis results with columns:
        - Feature: Name of categorical feature
        - Unique Values: Number of unique categories
        - Most Common: Most frequent category
        - Most Common %: Percentage of most frequent category
        - Least Common: Least frequent category
        - Least Common %: Percentage of least frequent category
        - Has Dominant: Whether feature has a dominant category
        - Rare Categories: Number of rare categories
    """
    results = []

    for col in df.select_dtypes(include=["object"]).columns:
        # Handle missing values by excluding them from percentage calculation
        value_counts = df[col].value_counts(normalize=True, dropna=True)
        n_categories = len(value_counts)
        max_freq = value_counts.max()
        min_freq = value_counts.min()

        # Check for dominant category (>threshold_dominant)
        dominant_category = max_freq > threshold_dominant

        # Check for rare categories (<threshold_rare)
        rare_categories = value_counts[value_counts < threshold_rare]

        results.append(
            {
                "Feature": col,
                "Unique Values": n_categories,
                "Most Common": value_counts.index[0],
                "Most Common %": max_freq * 100,
                "Least Common": value_counts.index[-1],
                "Least Common %": min_freq * 100,
                "Has Dominant": dominant_category,
                "Rare Categories": len(rare_categories),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(["Has Dominant", "Rare Categories"], ascending=False)
        .reset_index(drop=True)
    )
