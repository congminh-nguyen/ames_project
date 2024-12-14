import argparse
import json
import os
from pathlib import Path
from typing import Optional

import yaml
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
    # Check credentials in order of priority: CLI, environment, kaggle.json
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
    elif os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return
    else:
        kaggle_json_path = Path.home() / ".kaggle/kaggle.json"
        if kaggle_json_path.exists():
            with open(kaggle_json_path) as f:
                credentials = json.load(f)
            os.environ["KAGGLE_USERNAME"] = credentials.get("username")
            os.environ["KAGGLE_KEY"] = credentials.get("key")
        else:
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
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Error authenticating with Kaggle API: {e}")
        print("Please ensure you have the kaggle package installed: pip install kaggle")
        print(
            "And have valid credentials in ~/.kaggle/kaggle.json with correct permissions (chmod 600)"
        )
        return

    try:
        download_path = download_path.expanduser().resolve()
        download_path.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset: {dataset} to {download_path}")
        api.dataset_download_files(dataset, path=str(download_path), unzip=True)
        print(f"Dataset downloaded and extracted to {download_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and dataset identifier")


def load_config() -> dict:
    """
    Loads and returns the configuration from config.yaml file.

    Returns:
        dict: The loaded configuration dictionary, or empty dict if loading fails
    """
    try:
        # Get the package root directory by going up until we reach root
        current_dir = Path(__file__).resolve().parent
        while current_dir.name:  # Keep going up until we reach root
            config_path = current_dir / "config.yaml"
            if config_path.exists():
                break
            current_dir = current_dir.parent

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path) as file:
            config = yaml.safe_load(file)
            if not isinstance(config, dict):
                raise ValueError("Config file must contain a YAML dictionary/mapping")
            return config

    except Exception as e:
        print(f"Error loading config file: {e}")
        print("Please ensure config.yaml exists in the project root directory")
        return {}


def main() -> None:
    try:
        # Load configuration
        config = load_config()
        if not config:
            return

        # Get download path from config
        kaggle_config = config.get("kaggle", {})
        default_download_path = kaggle_config.get("download_path")
        default_dataset = kaggle_config.get("dataset")

        if not default_download_path:
            print("Error: download_path not found in config")
            return

        # Set up argument parser with defaults from config
        parser = argparse.ArgumentParser(description="Download Kaggle Dataset")
        parser.add_argument("-u", "--username", type=str, help="Kaggle username")
        parser.add_argument("-k", "--key", type=str, help="Kaggle API key")
        parser.add_argument(
            "-d",
            "--dataset",
            type=str,
            default=default_dataset,
            help="Kaggle dataset identifier (defaults to config value)",
        )
        args = parser.parse_args()

        if not args.dataset:
            print("Error: dataset not provided in arguments or config")
            return

        # Validate and set up credentials
        try:
            get_kaggle_credentials(args.username, args.key)
        except ValueError as e:
            print(e)
            print("To set up Kaggle credentials:")
            print("1. Create an account on kaggle.com")
            print("2. Go to Account -> Create New API Token")
            print("3. Place the downloaded kaggle.json in ~/.kaggle/")
            print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
            return

        # Download using config path and provided/default dataset
        download_data(args.dataset, Path(default_download_path))

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
