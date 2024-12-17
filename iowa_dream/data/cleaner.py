from typing import Any, Dict, List

import pandas as pd


def simple_fill_missing_by_keywords(
    df: pd.DataFrame, keywords: List[str]
) -> pd.DataFrame:
    """
    Fill missing values in columns containing any of the given keywords.
    For categorical columns, fills with 'NONE'.
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
            df_filled[col] = df[col].fillna("NONE")
        else:
            df_filled[col] = df[col].fillna(0)

    return df_filled


def garage_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute garage-related features in a specific order:
    1. Redenote 'No Garage' garage_type as 'NONE'
    2. Fill missing garage_type with 'NONE'
    3. Fix typo in garage_year_blt
    4. Fill missing garage_year_blt with year_blt
    5. Fill missing garage quality metrics with 'NONE'
    6. Fill missing numeric values with 0 only for houses with no garage or detached garage

    Args:
        df: DataFrame containing garage-related columns

    Returns:
        DataFrame with imputed garage values
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # 1. Redenote 'No Garage' garage_type as 'NONE'
    df.loc[df["garage_type"] == "No Garage", "garage_type"] = "NONE"

    # 2. Fill missing garage_type with 'NONE'
    df["garage_type"] = df["garage_type"].fillna("NONE")

    # 3. Fix typo in garage_year_blt from 2207 to 2007
    df.loc[df["garage_year_blt"] == 2207, "garage_year_blt"] = 2007

    # 4. Fill missing garage_year_blt with year_blt
    df.loc[df["garage_year_blt"].isnull(), "garage_year_blt"] = df.loc[
        df["garage_year_blt"].isnull(), "year_blt"
    ]

    # 5. Fill missing garage quality metrics with 'NONE'
    for col in ["garage_finish", "garage_qu", "garage_cond"]:
        df[col] = df[col].fillna("NONE")

    # 6. For houses with no garage or detached garage, fill missing numeric values with 0
    mask = df["garage_type"].isin(["NONE", "Detchd"])
    for col in ["garage_cars", "garage_area"]:
        df.loc[mask & df[col].isnull(), col] = 0

    return df


def type_formatting(
    df: pd.DataFrame,
    ordinal_cols: List[str],
    ordinal_mappings: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert ordinal columns based on predefined mappings.

    Args:
        df: Input DataFrame
        ordinal_cols: List of ordinal column names
        ordinal_mappings: List of dictionaries containing value mappings for ordinal columns
            Each dict must have 'values' and 'mapping' keys

    Returns:
        DataFrame with converted ordinal columns
    """
    df_converted = df.copy()

    # Validate ordinal_mappings format
    if not all(
        isinstance(entry, dict) and "values" in entry and "mapping" in entry
        for entry in ordinal_mappings
    ):
        raise ValueError(
            "Each entry in ordinal_mappings must be a dictionary with 'values' and 'mapping' keys."
        )

    # Create lookup of value sets to mappings
    mappings = {
        frozenset(entry["values"]): entry["mapping"] for entry in ordinal_mappings
    }

    # Convert ordinal columns based on predefined mappings
    for col in ordinal_cols:
        if pd.api.types.is_integer_dtype(df_converted[col]):
            continue

        # Find and apply mapping if column values match a mapping set
        unique_values = frozenset(df_converted[col].unique())
        if unique_values in mappings:
            df_converted[col] = df_converted[col].map(mappings[unique_values])

    return df_converted
