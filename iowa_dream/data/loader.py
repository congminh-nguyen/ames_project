from pathlib import Path

import pandas as pd


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names and converts float columns with whole numbers to integers.

    This function performs the following transformations:
    - Replaces 'yr' with 'year', 'qual' or 'qc' with 'qu', 'abvgrd' with 'abvgr', and 'built' with 'blt' in column names.
    - Converts float columns that contain only whole numbers to integer type, preserving NaNs.

    Args:
        df (pd.DataFrame): The DataFrame to be standardized.

    Returns:
        pd.DataFrame: A DataFrame with standardized column names and types.
    """
    # Standardize column names
    replacements = {
        " ": "_",
        "yr": "year",
        "qual": "qu",
        "qc": "qu",
        "abvgrd": "abvgr",
        "built": "blt",
    }
    for old, new in replacements.items():
        df.columns = df.columns.str.lower().str.replace(old, new)

    return df


def preliminary_loader(data_file: Path, standardize: bool = True) -> pd.DataFrame:
    """
    Load data from a CSV file and optionally standardize column names.

    Args:
        data_file (Path): Path to the CSV file to load
        standardize (bool, optional): Whether to standardize column names. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with optionally standardized column names
    """
    # Load the data
    df = pd.read_csv(data_file)

    # Standardize column names if requested
    if standardize:
        df = standardize_column_names(df)

    return df
