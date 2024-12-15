import filecmp  # For file comparison
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


def is_duplicate_file(existing_file: Path, new_file_path: Path) -> bool:
    """
    Check if an existing file is identical to a newly created file.
    """
    if existing_file.exists():
        return filecmp.cmp(existing_file, new_file_path, shallow=False)
    return False


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

    # Filter out the outliers where gr_liv_area > 4000 and saleprice < 200000
    df = df[~((df.gr_liv_area > 4000) & (df.saleprice < 200000))]

    # Impute missing values
    df = simple_fill_missing_by_keywords(
        df, ["pool", "bsmt", "fence", "fireplace", "alley", "misc_feature", "mas_vnr"]
    )
    df["electrical"] = df["electrical"].fillna("SBrkr")
    df = garage_imputer(df)

    # Extract feature column names for different data types
    data_dict = load_config()["data_dict"]
    ordinal = data_dict["ordinal_columns"]["columns"]
    nominal = data_dict["nominal_columns"]["columns"]
    continuous = data_dict["continuous_columns"]["columns"]
    discrete = data_dict["discrete_columns"]["columns"]

    df = type_formatting(
        df,
        discrete_cols=discrete,
        continuous_cols=continuous,
        nominal_cols=nominal,
        ordinal_cols=ordinal,
        ordinal_mappings=load_config().get("ordinal_mappings", []),
    )

    # Create cleaned directory if it doesn't exist
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    # Save cleaned data only if it doesn't already exist or is different
    if cleaned_path.exists():
        temp_cleaned_path = cleaned_dir / "temp_cleaned_AmesHousing.csv"
        df.to_csv(temp_cleaned_path, index=False)
        if is_duplicate_file(cleaned_path, temp_cleaned_path):
            print(
                f"The existing cleaned file at {cleaned_path} is identical. No changes made."
            )
            temp_cleaned_path.unlink()
        else:
            print(
                f"Updated cleaned file detected. Saving new cleaned file to {cleaned_path}"
            )
            temp_cleaned_path.rename(cleaned_path)
    else:
        df.to_csv(cleaned_path, index=False)
        print(f"Cleaned data saved to {cleaned_path}")


if __name__ == "__main__":
    main()
