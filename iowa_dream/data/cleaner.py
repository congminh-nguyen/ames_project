from typing import List

import pandas as pd


def convert_numerical_column_types(
    df: pd.DataFrame, discrete_cols: List[str], continuous_cols: List[str]
) -> pd.DataFrame:
    """
    Convert column types based on their nature: discrete or continuous.
    For continuous columns that are integers, convert to float.
    For discrete columns that are floats, convert to integer.
    Ignore NaN values during conversion.

    Args:
        df: pandas DataFrame
        discrete_cols: list of discrete column names
        continuous_cols: list of continuous column names
    Returns:
        DataFrame with converted column types
    """
    df_converted = df.copy()

    for col in continuous_cols:
        if pd.api.types.is_integer_dtype(df_converted[col].dropna()):
            df_converted[col] = df_converted[col].astype(float)

    for col in discrete_cols:
        if pd.api.types.is_float_dtype(df_converted[col].dropna()):
            df_converted[col] = df_converted[col].astype(
                "Int64"
            )  # Use 'Int64' to handle NaNs

    return df_converted


def simple_fill_missing_by_keywords(
    df: pd.DataFrame, keywords: List[str]
) -> pd.DataFrame:
    """
    Fill missing values in columns containing any of the given keywords.
    For categorical columns, fills with 'NA'.
    For numeric columns, fills with 0.

    Args:
        df: pandas DataFrame
        keywords: list of strings to match in column names
    Returns:
        DataFrame with filled missing values
    """
    # Make a copy to avoid modifying original
    df_filled = df.copy()

    # Get columns containing any of the keywords
    cols_to_fill = []
    for keyword in keywords:
        cols_to_fill.extend(
            [col for col in df.columns if keyword.lower() in col.lower()]
        )
    cols_to_fill = list(set(cols_to_fill))  # Remove duplicates

    for col in cols_to_fill:
        # Fill based on data type
        if df[col].dtype == "object":
            df_filled[col] = df[col].fillna("NA")
        else:
            df_filled[col] = df[col].fillna(0)

    return df_filled
