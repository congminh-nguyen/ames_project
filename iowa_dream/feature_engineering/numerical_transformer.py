from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted


class WinsorizedRobustScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer that applies winsorization followed by robust scaling.
    Winsorization clips values outside specified percentile range to reduce impact of outliers.
    RobustScaler then scales features using statistics robust to outliers.

    Compatible with sklearn.compose.ColumnTransformer and sklearn.pipeline.Pipeline.
    """

    def __init__(self, range_min: float = 1, range_max: float = 99) -> None:
        """
        Parameters
        ----------
        range_min : float, default=1
            Lower percentile for winsorization (between 0 and 100)
        range_max : float, default=99
            Upper percentile for winsorization (between 0 and 100)
        """
        if not 0 <= range_min < range_max <= 100:
            raise ValueError(
                "range_min must be less than range_max and both must be between 0 and 100"
            )

        self.range_min = range_min
        self.range_max = range_max
        self.robust_scaler = RobustScaler()

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "WinsorizedRobustScaler":
        """
        Compute winsorization bounds and fit the RobustScaler

        Parameters
        ----------
        X : pd.DataFrame
            Training data. Can be array or DataFrame when used directly,
            will receive array when used in ColumnTransformer
        y : Optional[Union[pd.Series, np.ndarray]], default=None
            Target values

        Returns
        -------
        self : object
            Returns self
        """
        # Convert to DataFrame if array
        X = pd.DataFrame(X)

        # Calculate percentile bounds for each feature
        self.lower_bounds_ = np.percentile(X, self.range_min, axis=0)
        self.upper_bounds_ = np.percentile(X, self.range_max, axis=0)

        # Store number of features for validation
        self.n_features_in_ = X.shape[1]

        # Winsorize data
        X_winsorized = np.clip(X, self.lower_bounds_, self.upper_bounds_)

        # Fit robust scaler
        self.robust_scaler.fit(X_winsorized)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorize and scale the data

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform. Can be array or DataFrame when used directly,
            will receive array when used in ColumnTransformer

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with scaled values
        """
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_", "n_features_in_"])

        # Convert to DataFrame if array
        X = pd.DataFrame(X)

        # Validate shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but WinsorizedRobustScaler "
                f"was fitted with {self.n_features_in_} features"
            )

        # Winsorize
        X_winsorized = np.clip(X, self.lower_bounds_, self.upper_bounds_)

        # Scale
        X_scaled = self.robust_scaler.transform(X_winsorized)

        return pd.DataFrame(X_scaled, columns=X.columns)
