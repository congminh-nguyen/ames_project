import sys
from pathlib import Path

# Add the root directory (AMES_PROJECT) to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from iowa_dream.data.cleaner import (
    garage_imputer,
    simple_fill_missing_by_keywords,
    type_formatting,
)
from iowa_dream.data.importer import load_config
from iowa_dream.data.loader import preliminary_loader


def main():
    try:
        config = load_config()
    except ModuleNotFoundError as e:
        print("Error: Could not load config or module not found.")
        print(str(e))
        sys.exit(1)

    # Resolve paths relative to project root
    download_path = project_root / config["kaggle"]["download_path"] / "AmesHousing.csv"
    cleaned_dir = project_root / config["kaggle"]["cleaned_path"]
    cleaned_path = cleaned_dir / "cleaned_AmesHousing.csv"

    print(f"Download path: {download_path}")
    print(f"Cleaned path: {cleaned_path}")

    # Load the data
    df = preliminary_loader(download_path)

    # Data Cleaning Process
    # Step 1: Outlier Removal
    # Remove outliers where the ground living area is greater than 4000 square feet. These are considered anomalies
    # that could skew the analysis.
    df = df[~(df.gr_liv_area > 4000)]
    # Step 2: Missing Value Imputation
    # Use keyword-based imputation
    df = simple_fill_missing_by_keywords(df, ["bsmt", "fireplace", "mas_vnr"])

    # Specifically fill missing values in the 'electrical' column with 'SBrkr',
    # which is the most common electrical system in the dataset.
    df["electrical"] = df["electrical"].fillna("SBrkr")
    # Use a specialized imputer for garage-related columns to handle missing
    # values based on domain-specific logic.
    df = garage_imputer(df)

    # Remove inconsistent years
    df = df[~(df["year_sold"] < df["year_blt"])]
    df = df[df["year_sold"] >= df["year_remod/add"]]

    # Drop features based on correlation analysis and highly correlated features
    # Get all columns from cleaned_data_dict
    keep_columns = (
        config["cleaned_data_dict"]["ordinal"]["columns"]
        + config["cleaned_data_dict"]["nominal"]["columns"]
        + config["cleaned_data_dict"]["discrete"]["columns"]
        + config["cleaned_data_dict"]["continuous"]["columns"]
        + ["saleprice", "pid"]
    )

    # Drop any column not in keep_columns
    columns_to_drop = [col for col in df.columns if col not in keep_columns] + [
        "mas_vnr_area"
    ]
    df = df.drop(columns=columns_to_drop)

    # Step 3: Data Type Formatting
    # Load the data dictionary from the configuration to extract feature column
    # names categorized by their data types: ordinal, nominal, continuous, and discrete.
    ordinal_mappings = load_config()["ordinal_mappings"]

    # Format the data types of the columns according to their categories.
    # This includes converting columns to appropriate data types and applying
    # ordinal mappings where necessary.
    df = type_formatting(
        df,
        [
            "exter_qu",
            "bsmt_qu",
            "bsmt_exposure",
            "heating_qu",
            "kitchen_qu",
            "fireplace_qu",
        ],
        ordinal_mappings,
    )

    # Create cleaned directory if it doesn't exist
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    # Overwrite the cleaned data file
    df.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved to {cleaned_path}")


if __name__ == "__main__":
    main()
