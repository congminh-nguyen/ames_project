from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NominalTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, bins: Optional[List[int]] = None, labels: Optional[List[str]] = None
    ) -> None:
        # Define default bins and labels for age-based ranges
        self.bins = bins if bins else [0, 10, 30, 50, 150]
        self.labels = labels if labels else ["0-10", "11-30", "31-50", "51+"]

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "NominalTransformer":
        # No fitting necessary for this transformer
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Ensure X is a DataFrame
        X = pd.DataFrame(X).copy()

        # Check required columns
        required_columns = {
            "year_blt",
            "year_sold",
            "year_remod/add",
            "exterior_1st",
            "exterior_2nd",
        }
        if not required_columns.issubset(X.columns):
            raise ValueError(
                f"Input DataFrame must contain the following columns: {required_columns}"
            )

        # 1. Replace year_remod/add: Binary indicator for remodeling
        # Changed to check if remod year equals sale year instead of build year
        X["year_remod/add"] = (X["year_remod/add"] > X["year_blt"]).astype(int)

        # 2. Replace year_blt: Bin the age of the house at the time of sale
        X["year_blt"] = X["year_sold"] - X["year_blt"]  # Calculate age at sale
        X["year_blt"] = pd.cut(
            X["year_blt"], bins=self.bins, labels=self.labels, right=False
        )

        # 3. Replace exterior_2nd: Binary indicator for matching exterior materials
        X["exterior_2nd"] = (X["exterior_1st"] == X["exterior_2nd"]).astype(int)

        return X


class OrdinalMerger(BaseEstimator, TransformerMixin):
    def __init__(self, min_obs=10):
        """
        Custom transformer for ordinal variables.

        Parameters:
        - min_obs: Minimum number of observations required for each category. If not met,
                   categories will be merged with adjacent categories to maintain ordinality.
        """
        self.min_obs = min_obs
        self.mapping_ = {}

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters:
        - X: A pandas DataFrame containing ordinal variables.
        - y: Ignored (compatibility with scikit-learn pipelines).

        Returns:
        - self: Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        self.mapping_ = {}

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

    def transform(self, X):
        """
        Transform the data by merging categories as determined during fitting.

        Parameters:
        - X: A pandas DataFrame containing ordinal variables.

        Returns:
        - X_transformed: Transformed DataFrame with merged categories.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        X_transformed = X.copy()

        for col in X.columns:
            if col in self.mapping_:
                X_transformed[col] = X[col].map(self.mapping_[col])

        return X_transformed
