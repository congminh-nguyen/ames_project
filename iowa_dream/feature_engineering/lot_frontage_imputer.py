# Import necessary libraries
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Define the GroupMedianImputer class
class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_cols: List[str], target_col: str) -> None:
        """
        Custom imputer to fill missing values using group-wise median.

        Parameters:
        group_cols: list of str
            Columns to group by for calculating medians.
        target_col: str
            Column to impute missing values in.
        """
        self.group_cols = group_cols
        self.target_col = target_col

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "GroupMedianImputer":
        """
        Fit the imputer. No operation needed here, as medians are calculated dynamically during transform.

        Parameters:
        X: pandas.DataFrame
            DataFrame containing the data.
        y: Ignored
            Not used, exists for compatibility with scikit-learn pipelines.

        Returns:
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Separate missing and non-missing data
        missing_mask = X[self.target_col].isna()
        non_missing_data = X.loc[~missing_mask]
        missing_data = X.loc[missing_mask]

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
        """
        Fit the imputer and transform the data.

        Parameters:
        X: pandas.DataFrame
            DataFrame containing the data.
        y: Ignored
            Not used, exists for compatibility with scikit-learn pipelines.

        Returns:
        X_transformed: pandas.DataFrame
            DataFrame with missing values imputed.
        """
        return self.fit(X, y).transform(X)
