from typing import Optional

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
        self.mapping_: dict = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OrdinalMerger":
        """Fit the transformer to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features containing ordinal variables
        y : Optional[pd.Series], default=None
            Target variable (unused)

        Returns
        -------
        self : OrdinalMerger
            Returns self

        Raises
        ------
        ValueError
            If input is not a pandas DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        self.mapping_ = {}

        # Handle each column separately
        for col in X.columns:
            value_counts = X[col].value_counts().sort_index()
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

            self.mapping_[col] = mapping

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by merging categories.

        Parameters
        ----------
        X : pd.DataFrame
            Input features containing ordinal variables

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with merged categories

        Raises
        ------
        ValueError
            If input is not a pandas DataFrame or if transformer has not been fitted
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not hasattr(self, "mapping_"):
            raise ValueError("OrdinalMerger has not been fitted yet.")

        X_transformed = X.copy()

        for col in X.columns:
            if col in self.mapping_:
                X_transformed[col] = X[col].map(self.mapping_[col])

        return X_transformed

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features containing ordinal variables
        y : Optional[pd.Series], default=None
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
        self.mapping_: dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer by identifying infrequent categories.

        Parameters
        ----------
        X : pd.DataFrame
            Input features containing nominal variables
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        self.mapping_ = {}

        for col in X.columns:
            # Skip year and month columns
            if "year" in col.lower() or col == "mo_sold":
                continue

            value_counts = X[col].value_counts()
            # Categories with counts below threshold get mapped to 'Other'
            infrequent = value_counts[value_counts < self.min_obs].index
            mapping = {
                cat: "Other" if cat in infrequent else cat for cat in value_counts.index
            }
            self.mapping_[col] = mapping

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by grouping infrequent categories.

        Parameters
        ----------
        X : pd.DataFrame
            Input features containing nominal variables

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with grouped categories
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not hasattr(self, "mapping_"):
            raise ValueError("NominalGrouper has not been fitted yet.")

        X_transformed = X.copy()

        for col in X.columns:
            if col in self.mapping_:
                X_transformed[col] = X[col].map(self.mapping_[col])

        return X_transformed
