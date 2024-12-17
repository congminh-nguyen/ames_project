from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Impute missing values using group-wise median.

    Parameters
    ----------
    group_cols : List[str]
        Columns to group by for calculating medians.
    target_col : str
        Column to impute missing values in.

    Attributes
    ----------
    group_cols : List[str]
        Columns to group by for calculating medians.
    target_col : str
        Column to impute missing values in.
    """

    def __init__(self, group_cols: List[str], target_col: str) -> None:
        self.group_cols = group_cols
        self.target_col = target_col

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "GroupMedianImputer":
        """Fit the imputer.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit.
        y : Optional[Union[pd.Series, np.ndarray]], default=None
            Target values. Not used, present for API consistency.

        Returns
        -------
        self : GroupMedianImputer
            Returns self.

        Raises
        ------
        ValueError
            If input is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by imputing missing values.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.

        Returns
        -------
        pd.DataFrame
            Data with missing values imputed.

        Raises
        ------
        ValueError
            If input is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Create copy to avoid modifying original data
        X_copy = X.copy()

        # Separate missing and non-missing data
        missing_mask = X_copy[self.target_col].isna()
        non_missing_data = X_copy.loc[~missing_mask]
        missing_data = X_copy.loc[missing_mask]

        # Calculate medians for non-missing data, grouped by group_cols
        group_medians = (
            non_missing_data.groupby(self.group_cols)[self.target_col]
            .median()
            .reset_index()
            .rename(columns={self.target_col: "median"})
        )

        # Merge medians onto the missing data
        missing_data = missing_data.merge(group_medians, on=self.group_cols, how="left")

        # Fill missing values with group-wise median or fallback to overall median
        global_median = non_missing_data[self.target_col].median()
        missing_data[self.target_col] = (
            missing_data[self.target_col]
            .fillna(missing_data["median"])
            .fillna(global_median)
        )

        # Drop the auxiliary median column
        missing_data = missing_data.drop(columns=["median"])

        # Combine non-missing and imputed missing data
        result = pd.concat([non_missing_data, missing_data]).sort_index()
        return result

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> pd.DataFrame:
        """Fit the imputer and transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit and transform.
        y : Optional[Union[pd.Series, np.ndarray]], default=None
            Target values. Not used, present for API consistency.

        Returns
        -------
        pd.DataFrame
            Data with missing values imputed.
        """
        return self.fit(X, y).transform(X)
