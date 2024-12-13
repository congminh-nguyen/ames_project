from pathlib import Path

import pandas as pd


def preliminary_load_and_clean_data(data_file: Path) -> pd.DataFrame:
    """
    Load CSV data and standardize column names.

    Args:
        data_file (Path): Path to the CSV file to load

    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    df = pd.read_csv(data_file.resolve())

    # Create a mapping of old column names to new standardized names
    column_mapping = {}
    for col in df.columns:
        # Convert to lowercase
        new_name = col.lower()
        # Replace spaces with underscores
        new_name = new_name.replace(" ", "_")
        # Replace special characters with underscores
        new_name = "".join(c if c.isalnum() or c == "_" else "_" for c in new_name)
        # Remove consecutive underscores
        while "__" in new_name:
            new_name = new_name.replace("__", "_")
        # Remove trailing underscores
        new_name = new_name.rstrip("_")

        column_mapping[col] = new_name

    # Rename the columns using the mapping
    df = df.rename(columns=column_mapping)

    return df
