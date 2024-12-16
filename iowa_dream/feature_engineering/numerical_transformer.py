import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class RobustScalerWithIndicator(BaseEstimator, TransformerMixin):
    def __init__(
        self, add_indicator: bool = True, output_dataframe: bool = True
    ) -> None:
        self.add_indicator = add_indicator
        self.output_dataframe = output_dataframe

    def fit(self, X, y=None):
        X = self._validate_input(X)
        self.scalers_ = {}
        for col in X.columns:
            values = X[col].values
            center = np.median(values)
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            scale = q3 - q1
            scale = scale if scale != 0 else 1.0
            self.scalers_[col] = {"center": center, "scale": scale}
        return self

    def transform(self, X):
        if not hasattr(self, "scalers_"):
            raise NotFittedError(
                "This RobustScalerWithIndicator instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )
        X = self._validate_input(X)

        # Check for missing columns
        missing_cols = set(self.scalers_.keys()) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input: {missing_cols}")

        X_transformed = pd.DataFrame(index=X.index)
        for (
            col
        ) in (
            self.scalers_.keys()
        ):  # Iterate through fitted columns instead of input columns
            scaler_params = self.scalers_[col]
            new_col = X[col].astype(float).copy()
            center = scaler_params["center"]
            scale = scaler_params["scale"]
            new_col = (new_col - center) / scale
            X_transformed[col] = new_col
            if self.add_indicator:
                X_transformed[f"{col}_zero_indicator"] = (X[col] == 0).astype(int)
        return X_transformed if self.output_dataframe else X_transformed.values

    def inverse_transform(self, X):
        if not hasattr(self, "scalers_"):
            raise NotFittedError(
                "This RobustScalerWithIndicator instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )
        X = self._validate_input(X)

        # Check for missing columns (excluding indicator columns)
        original_cols = set(self.scalers_.keys())
        input_cols = {col for col in X.columns if not col.endswith("_zero_indicator")}
        missing_cols = original_cols - input_cols
        if missing_cols:
            raise ValueError(f"Missing columns in input: {missing_cols}")

        X_inverse = pd.DataFrame(index=X.index)
        for col in self.scalers_.keys():
            scaler_params = self.scalers_[col]
            new_col = X[col].astype(float).copy()
            center = scaler_params["center"]
            scale = scaler_params["scale"]
            new_col = (new_col * scale) + center
            X_inverse[col] = new_col
        return X_inverse if self.output_dataframe else X_inverse.values

    def _validate_input(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        else:
            raise ValueError("Input data must be a pandas DataFrame or numpy ndarray.")
