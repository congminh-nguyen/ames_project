from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class WinsorizedRobustScaler(BaseEstimator, TransformerMixin):
    """
    A transformer compatible with ColumnTransformer that clips data to
    specified percentiles (winsorization) and then applies robust scaling.

    Parameters
    ----------
    range_min : float, default=1
        Lower percentile for winsorization (between 0 and 100).
    range_max : float, default=99
        Upper percentile for winsorization (between 0 and 100).
    """

    def __init__(self, range_min: float = 1, range_max: float = 99) -> None:
        if not 0 <= range_min < range_max <= 100:
            raise ValueError("range_min must be < range_max, both between 0 and 100.")
        self.range_min = range_min
        self.range_max = range_max

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "WinsorizedRobustScaler":
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : np.ndarray
            The input data to fit (2D array or DataFrame).
        y : Optional[np.ndarray], default=None
            Target values. Not used.

        Returns
        -------
        self : WinsorizedRobustScaler
            Fitted transformer.
        """
        X = pd.DataFrame(X)  # Ensure input is a DataFrame for percentile calculation
        self.lower_bounds_ = X.quantile(self.range_min / 100.0).values
        self.upper_bounds_ = X.quantile(self.range_max / 100.0).values
        X_clipped = X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        self.medians_ = X_clipped.median().values
        self.iqr_ = X_clipped.quantile(0.75).values - X_clipped.quantile(0.25).values
        self.iqr_ = np.where(self.iqr_ == 0, 1, self.iqr_)  # Handle zero IQR
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data by winsorizing and robust scaling.

        Parameters
        ----------
        X : np.ndarray
            The input data to transform (2D array or DataFrame).

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_", "medians_", "iqr_"])
        X = pd.DataFrame(X)  # Ensure input is a DataFrame for compatibility
        X_clipped = X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)

        # Combine handling of zero IQR and constant features
        iqr_safe = np.where(self.iqr_ == 0, 1, self.iqr_)
        X_scaled = (X_clipped - self.medians_) / iqr_safe

        # Set constant features to 0 in the scaled data
        constant_features = (self.iqr_ == 0) | (X.nunique() == 1).values
        X_scaled.loc[:, constant_features] = 0

        return X_scaled.values

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray
            The input data to fit and transform.
        y : Optional[np.ndarray], default=None
            Target values. Not used.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        return self.fit(X, y).transform(X)
