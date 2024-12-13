import hashlib

import numpy as np
import pandas as pd


def create_sample_split(
    df: pd.DataFrame, id_col: str, training_frac: float = 0.8
) -> pd.DataFrame:
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_col : str
        Name of ID column to use for splitting
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    if df[id_col].dtype == np.int64:
        modulo = df[id_col] % 100
    else:
        modulo = df[id_col].apply(
            lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 100
        )

    df["sample"] = np.where(modulo < training_frac * 100, "train", "test")

    return df
