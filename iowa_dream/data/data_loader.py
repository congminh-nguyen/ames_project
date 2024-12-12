import argparse
import json
import os
from pathlib import Path
from typing import Optional

from kaggle.api.kaggle_api_extended import KaggleApi


def get_kaggle_credentials(username: Optional[str], key: Optional[str]) -> None:
    """
    Retrieves Kaggle credentials from command-line arguments, environment variables,
    or the default kaggle.json file.

    Args:
        username (Optional[str]): Kaggle username provided via CLI.
        key (Optional[str]): Kaggle API key provided via CLI.

    Raises:
        ValueError: If no valid credentials are found.
    """
    # Priority 1: Command-line arguments
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        return

    # Priority 2: Environment variables
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return

    # Priority 3: kaggle.json
    kaggle_json_path = Path.home() / ".kaggle/kaggle.json"
    if kaggle_json_path.exists():
        with open(kaggle_json_path) as f:
            credentials = json.load(f)
        os.environ["KAGGLE_USERNAME"] = credentials.get("username")
        os.environ["KAGGLE_KEY"] = credentials.get("key")
        return

    # If no valid credentials are found
    raise ValueError(
        "Kaggle credentials not found. Provide them via command-line arguments, "
        "environment variables, or a kaggle.json file."
    )


def download_data(dataset: str, download_path: Path) -> None:
    """
    Downloads the specified Kaggle dataset to the provided download path.

    Args:
        dataset (str): Kaggle dataset identifier (e.g., "username/dataset-name").
        download_path (Path): Directory to save the dataset.
    """
    api = KaggleApi()
    api.authenticate()

    download_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset: {dataset} to {download_path}")
    api.dataset_download_files(dataset, path=str(download_path), unzip=True)
    print(f"Dataset downloaded and extracted to {download_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle Dataset")
    parser.add_argument("-u", "--username", type=str, help="Kaggle username (optional)")
    parser.add_argument("-k", "--key", type=str, help="Kaggle API key (optional)")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Kaggle dataset identifier"
    )
    parser.add_argument(
        "--download-path",
        type=str,
        default="./iowa_dream/_rawfile",
        help="Path to download the dataset",
    )
    args = parser.parse_args()

    # Validate and set up credentials
    try:
        get_kaggle_credentials(args.username, args.key)
    except ValueError as e:
        print(e)
        return

    # Download the dataset
    download_path = Path(args.download_path)
    download_data(args.dataset, download_path)


if __name__ == "__main__":
    main()
