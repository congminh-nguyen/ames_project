from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OrdinalMerger(BaseEstimator, TransformerMixin):
    def __init__(self, min_obs: int = 10) -> None:
        """Initialize OrdinalMerger.

        Parameters
        ----------
        min_obs : int, default=10
            Minimum number of observations required for each category. If not met,
            categories will be merged with adjacent categories to maintain ordinality.
        """
        self.min_obs = min_obs
        self.mapping_: Dict[str, Dict[str, str]] = {}

    def fit(
        self, X: Any, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "OrdinalMerger":
        """Fit the transformer to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features containing ordinal variables
        y : Optional[Union[pd.Series, np.ndarray]], default=None
            Target variable (unused)

        Returns
        -------
        self : OrdinalMerger
            Returns self
        """
        # Convert to DataFrame if array
        X = pd.DataFrame(X)

        self.n_features_in_ = X.shape[1]
        self.mapping_ = {}

        # Handle each column separately
        for col_idx in range(X.shape[1]):
            value_counts = X.iloc[:, col_idx].value_counts().sort_index()
            categories = value_counts.index.tolist()
            counts = value_counts.values.copy()

            # Initialize mapping
            mapping = {cat: cat for cat in categories}

            while True:
                # Find smallest category below threshold
                small_cats = [
                    i for i, count in enumerate(counts) if count < self.min_obs
                ]
                if not small_cats:
                    break

                idx = small_cats[0]  # Process smallest category first

                if len(counts) <= 1:  # No more merging possible
                    break

                # Always merge with adjacent category that has more observations
                if idx == 0:  # First category - can only merge up
                    merge_idx = idx + 1
                elif idx == len(counts) - 1:  # Last category - can only merge down
                    merge_idx = idx - 1
                else:
                    # Choose direction with more observations
                    merge_idx = (
                        idx + 1 if counts[idx + 1] >= counts[idx - 1] else idx - 1
                    )

                # Perform merge
                target_cat = categories[merge_idx]
                source_cat = categories[idx]
                counts[merge_idx] += counts[idx]

                # Update mapping
                mapping[source_cat] = target_cat

                # Remove merged category
                categories.pop(idx)
                counts = np.delete(counts, idx)

                # Update any previous mappings that pointed to the removed category
                for k, v in mapping.items():
                    if v == source_cat:
                        mapping[k] = target_cat

            self.mapping_[str(col_idx)] = mapping

        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """Transform the data by merging categories.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features containing ordinal variables

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with merged categories
        """
        # Convert to DataFrame if array
        X = pd.DataFrame(X)

        if not hasattr(self, "mapping_"):
            raise ValueError("OrdinalMerger has not been fitted yet.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but OrdinalMerger was fitted with {self.n_features_in_} features"
            )

        X_transformed = X.copy()

        for col_idx in range(X.shape[1]):
            if str(col_idx) in self.mapping_:
                X_transformed.iloc[:, col_idx] = X.iloc[:, col_idx].map(
                    self.mapping_[str(col_idx)]
                )

        return X_transformed

    def fit_transform(
        self, X: Any, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> pd.DataFrame:
        """Fit and transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features containing ordinal variables
        y : Optional[Union[pd.Series, np.ndarray]], default=None
            Target variable (unused)

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with merged categories
        """
        return self.fit(X, y).transform(X)


class NominalGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_obs: int = 10) -> None:
        """Initialize NominalGrouper.

        Parameters
        ----------
        min_obs : int, default=10
            Minimum number of observations required for each category.
            Categories with fewer observations will be grouped into 'Other'.
        """
        self.min_obs = min_obs
        self.mapping_: Dict[str, Dict[str, str]] = {}

    def fit(
        self, X: Any, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "NominalGrouper":
        """Fit the transformer by identifying infrequent categories.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features containing nominal variables
        y : Optional[Union[pd.Series, np.ndarray]], default=None
            Target variable (unused)

        Returns
        -------
        self : NominalGrouper
            Returns self
        """
        # Convert to DataFrame if array
        X = pd.DataFrame(X)

        self.n_features_in_ = X.shape[1]
        self.mapping_ = {}

        for col_idx in range(X.shape[1]):
            value_counts = X.iloc[:, col_idx].value_counts()
            # Categories with counts below threshold get mapped to 'Other'
            infrequent = value_counts[value_counts < self.min_obs].index
            mapping = {
                cat: "Other" if cat in infrequent else cat for cat in value_counts.index
            }
            self.mapping_[str(col_idx)] = mapping

        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """Transform the data by grouping infrequent categories.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features containing nominal variables

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with grouped categories
        """
        # Convert to DataFrame if array
        X = pd.DataFrame(X)

        if not hasattr(self, "mapping_"):
            raise ValueError("NominalGrouper has not been fitted yet.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but NominalGrouper was fitted with {self.n_features_in_} features"
            )

        X_transformed = X.copy()

        for col_idx in range(X.shape[1]):
            if str(col_idx) in self.mapping_:
                X_transformed.iloc[:, col_idx] = X.iloc[:, col_idx].map(
                    self.mapping_[str(col_idx)]
                )

        return X_transformed
