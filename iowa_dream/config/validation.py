from typing import Optional

from pydantic import BaseModel, DirectoryPath, Field, FilePath


class KaggleConfig(BaseModel):
    api_key_path: Optional[FilePath] = Field(
        None, description="Path to Kaggle API credentials (kaggle.json)"
    )
    dataset: str = Field(
        ..., description="Kaggle dataset identifier (e.g., 'username/dataset-name')"
    )
    download_path: DirectoryPath = Field(
        ..., description="Directory where the dataset will be downloaded"
    )


class CommandLineArgs(BaseModel):
    username: Optional[str] = Field(
        None, description="Kaggle username (optional if kaggle.json exists)"
    )
    key: Optional[str] = Field(
        None, description="Kaggle API key (optional if kaggle.json exists)"
    )
    dataset: str = Field(
        ..., description="Kaggle dataset identifier (e.g., 'username/dataset-name')"
    )
    download_path: Optional[DirectoryPath] = Field(
        None, description="Custom path to download the dataset"
    )
