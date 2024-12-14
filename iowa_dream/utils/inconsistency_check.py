import pandas as pd


def report_inconsistent_years(
    df: pd.DataFrame, built_year_cols: list, year_sold_col: str
) -> pd.DataFrame:
    """
    Report rows with inconsistencies in built year compared to 2010 and year_sold column.

    Args:
        df (pd.DataFrame): Input DataFrame containing the built year columns and year_sold column
        built_year_cols (list): List of column names representing built years
        year_sold_col (str): Column name representing the year the property was sold

    Returns:
        pd.DataFrame: DataFrame with rows containing inconsistencies
    """
    final_year = 2010
    inconsistent_rows = pd.DataFrame()

    for col in built_year_cols:
        # Get rows with inconsistencies in the column
        inconsistent_rows_col = df[
            (df[col] > final_year) | (df[col] > df[year_sold_col])
        ][built_year_cols + [year_sold_col]]
        # Append inconsistent rows to the result DataFrame
        inconsistent_rows = pd.concat([inconsistent_rows, inconsistent_rows_col])

    # Drop duplicate rows if any
    inconsistent_rows = inconsistent_rows.drop_duplicates()

    return inconsistent_rows


def consistency_missing_type_area(df, area_column, type_column):
    """
    Compute the percentage of missing values in a type column for different conditions of an area column.

    Args:
        df: pandas DataFrame
        area_column: Name of the area column
        type_column: Name of the type column

    Returns:
        A dictionary with percentages of missing values in the type column for each condition of the area column.
    """
    total_missing_type = df[type_column].isnull().sum()

    # Percentage of missing type where area is 0
    missing_type_area_zero = df[
        (df[area_column] == 0) & (df[type_column].isnull())
    ].shape[0]
    percent_missing_area_zero = (missing_type_area_zero / total_missing_type) * 100

    # Percentage of missing type where area is missing
    missing_type_area_missing = df[
        (df[area_column].isnull()) & (df[type_column].isnull())
    ].shape[0]
    percent_missing_area_missing = (
        missing_type_area_missing / total_missing_type
    ) * 100

    # Percentage of missing type where area has other values
    missing_type_area_other = df[
        (df[area_column] != 0)
        & (df[area_column].notnull())
        & (df[type_column].isnull())
    ].shape[0]
    percent_missing_area_other = (missing_type_area_other / total_missing_type) * 100

    return {
        "percent_missing_area_zero": percent_missing_area_zero,
        "percent_missing_area_missing": percent_missing_area_missing,
        "percent_missing_area_other": percent_missing_area_other,
    }
