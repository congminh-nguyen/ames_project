from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class LotFrontageGroupMedianImputer(BaseEstimator, TransformerMixin):
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
        super().__init__()

    def get_feature_names_out(self, input_features=None):
        """Get output feature names.
        
        Parameters
        ----------
        input_features : list of str or None
            Input features.
            
        Returns
        -------
        list of str
            Output feature names.
        """
        return input_features

    def set_output(self, *, transform=None):
        """Set output container.
        
        Parameters
        ----------
        transform : {'default', 'pandas'}, default=None
            Configure output of transform and fit_transform.

            - 'default': Default output format of a transformer
            - 'pandas': DataFrame output
            - None: Transform configuration is unchanged
            
        Returns
        -------
        self
            Transformer instance.
        """
        if transform not in ['default', 'pandas', None]:
            raise ValueError(
                "Valid values for transform are 'default', 'pandas', None. "
                f"Got transform={transform!r}"
            )
        self._transform = transform
        return self

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "LotFrontageGroupMedianImputer":
        """Fit the imputer.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit.
        y : Optional[Union[pd.Series, np.ndarray]], default=None
            Target values. Not used, present for API consistency.

        Returns
        -------
        self : LotFrontageGroupMedianImputer
            Returns self.

        Raises
        ------
        ValueError
            If input is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        # Validate that required columns exist
        missing_cols = [col for col in self.group_cols + [self.target_col] if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
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

        # Validate that required columns exist
        missing_cols = [col for col in self.group_cols + [self.target_col] if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Create copy to avoid modifying original data
        X_copy = X.copy()

        # Calculate group medians only if there are missing values
        if X_copy[self.target_col].isna().any():
            # Calculate medians for each group
            group_medians = X_copy.groupby(self.group_cols)[self.target_col].transform('median')
            
            # Fill missing values with group medians first
            X_copy[self.target_col] = X_copy[self.target_col].fillna(group_medians)
            
            # If any values are still missing, fill with overall median
            if X_copy[self.target_col].isna().any():
                overall_median = X_copy[self.target_col].median()
                X_copy[self.target_col] = X_copy[self.target_col].fillna(overall_median)

        return X_copy

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