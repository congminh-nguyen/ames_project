from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import PowerTransformer, StandardScaler


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler_ = StandardScaler()
        self.power_transformer_ = PowerTransformer(method="yeo-johnson")
        self.columns_: List[str] = []

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "NumericalTransformer":
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        self.columns_ = X.columns.tolist()

        # First apply Yeo-Johnson transform
        self.power_transformer_.fit(X)

        # Then fit standard scaler
        X_transformed = self.power_transformer_.transform(X)
        self.scaler_.fit(X_transformed)

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if not hasattr(self, "columns_"):
            raise NotFittedError(
                "NumericalTransformer is not fitted yet. Call 'fit' first."
            )

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Check if all required columns are present
        missing_cols = set(self.columns_) - set(X.columns)
        if missing_cols:
            raise KeyError(list(missing_cols)[0])

        # First apply Yeo-Johnson transform
        X_transformed = self.power_transformer_.transform(X)

        # Then apply standard scaling
        X_transformed = self.scaler_.transform(X_transformed)

        return pd.DataFrame(X_transformed, columns=self.columns_, index=X.index)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
