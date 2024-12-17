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
    df = df.drop(
        [
            "bsmt_exposure",
            "bsmt_full_bath",
            "bsmt_half_bath",
            "bsmt_unf_sf",
            "bsmtfin_sf_2",
            "bsmtfin_type_2",
            "central_air",
            "condition_2",
            "enclosed_porch",
            "exter_qu",
            "exterior_2nd",
            "fence",
            "functional",
            "garage_area",
            "garage_cond",
            "garage_finish",
            "garage_type",
            "garage_year_blt",
            "house_style",
            "land_contour",
            "land_slope",
            "low_qu_fin_sf",
            "misc_feature",
            "misc_val",
            "ms_zoning",
            "open_porch_sf",
            "pool_area",
            "pool_qu",
            "roof_matl",
            "sale_condition",
            "screen_porch",
            "street",
            "totrms_abvgr",
            "utilities",
            "3ssn_porch",
            "bldg_type",
            "1st_flr_sf",
            "alley",
            "order",
            "bsmtfin_type_1",
        ],
        axis=1,
    )

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
            "garage_qu",
            "fireplace_qu",
            "kitchen_qu",
            "heating_qu",
            "bsmt_qu",
            "bsmt_cond",
            "lot_shape",
            "exter_cond",
            "paved_drive",
        ],
        ordinal_mappings,
    )

    # Step 4: Drop Unnecessary Features
    # Drop features that are deemed unnecessary or redundant as specified in the configuration.
    preliminary_dropped_features = config.get("preliminary_dropped_features", [])
    df = df.drop(columns=preliminary_dropped_features, errors="ignore")

    # Create cleaned directory if it doesn't exist
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    # Overwrite the cleaned data file
    df.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved to {cleaned_path}")


if __name__ == "__main__":
    main()
