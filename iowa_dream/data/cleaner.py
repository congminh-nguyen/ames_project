from typing import Any, Dict, List, Optional

import pandas as pd


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


def garage_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute garage-related features in a specific order:
    1. Redenote 'No Garage' garage_type as 'NA'
    2. Fill missing garage_type with 'NA'
    3. Fix typo in garage_year_blt
    4. Fill missing garage_year_blt with year_blt
    5. Fill missing garage quality metrics with 'NA'
    6. Fill missing numeric values with 0 only for houses with no garage or detached garage

    Args:
        df: DataFrame containing garage-related columns

    Returns:
        DataFrame with imputed garage values
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # 1. Redenote 'No Garage' garage_type as 'NA'
    df.loc[df["garage_type"] == "No Garage", "garage_type"] = "NA"

    # 2. Fill missing garage_type with 'NA'
    df["garage_type"] = df["garage_type"].fillna("NA")

    # 3. Fix typo in garage_year_blt from 2207 to 2007
    df.loc[df["garage_year_blt"] == 2207, "garage_year_blt"] = 2007

    # 4. Fill missing garage_year_blt with year_blt
    df.loc[df["garage_year_blt"].isnull(), "garage_year_blt"] = df.loc[
        df["garage_year_blt"].isnull(), "year_blt"
    ]

    # 5. Fill missing garage quality metrics with 'NA'
    for col in ["garage_finish", "garage_qu", "garage_cond"]:
        df[col] = df[col].fillna("NA")

    # 6. For houses with no garage or detached garage, fill missing numeric values with 0
    mask = df["garage_type"].isin(["NA", "Detchd"])
    for col in ["garage_cars", "garage_area"]:
        df.loc[mask & df[col].isnull(), col] = 0

    return df


def type_formatting(
    df: pd.DataFrame,
    discrete_cols: List[str],
    continuous_cols: List[str],
    nominal_cols: Optional[List[str]] = None,
    ordinal_cols: Optional[List[str]] = None,
    ordinal_mappings: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Convert column types based on their nature:
    - Convert continuous integer columns to float
    - Convert discrete float columns to integer
    - Convert nominal numeric columns to string category
    - Convert ordinal columns based on predefined mappings
    Ignore NaN values during conversion.
    """
    df_converted = df.copy()

    # Validate ordinal_mappings
    if ordinal_mappings is None:
        if ordinal_cols is not None:
            raise ValueError(
                "If ordinal_mappings is None, ordinal_cols should also be None or provide a list."
            )
        ordinal_mappings = []
    if not all(
        isinstance(entry, dict) and "values" in entry and "mapping" in entry
        for entry in ordinal_mappings
    ):
        raise ValueError(
            "Each entry in ordinal_mappings must be a dictionary with 'values' and 'mapping' keys."
        )

    # Convert continuous integer columns to float
    for col in continuous_cols:
        if pd.api.types.is_integer_dtype(df_converted[col]):
            df_converted[col] = df_converted[col].astype(float)

    # Convert discrete float columns to integer
    for col in discrete_cols:
        if pd.api.types.is_float_dtype(df_converted[col]):
            df_converted[col] = df_converted[col].astype(
                "Int64"
            )  # Use 'Int64' to handle NaNs

    # Convert nominal numeric columns to string category
    if nominal_cols:
        for col in nominal_cols:
            if pd.api.types.is_numeric_dtype(df_converted[col]):
                df_converted[col] = df_converted[col].astype(str)

    # Convert ordinal columns based on predefined mappings
    if ordinal_cols:
        mappings = {
            frozenset(entry["values"]): entry["mapping"] for entry in ordinal_mappings
        }
        for col in ordinal_cols:
            if pd.api.types.is_integer_dtype(df_converted[col]):
                continue

            # Drop NaN values to ensure correct mapping
            unique_values = frozenset(df_converted[col].unique())
            if unique_values in mappings:
                df_converted[col] = df_converted[col].map(mappings[unique_values])

    return df_converted
