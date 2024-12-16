from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropFeatures(BaseEstimator, TransformerMixin):
    """
    A transformer to drop specified columns from a DataFrame.

    Parameters:
    -----------
    col_drop : Optional[List[str]], default=None
        List of column names to drop from the DataFrame. If None, no columns are dropped.
    """

    def __init__(self, col_drop: Optional[List[str]] = None):
        self.col_drop = col_drop

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DropFeatures":
        """Fit method - no fitting required for dropping columns."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop specified columns from the DataFrame.

        Parameters:
        -----------
        X : pd.DataFrame
            Input DataFrame to transform.

        Returns:
        --------
        pd.DataFrame
            DataFrame with specified columns dropped.
        """
        X_copy = X.copy()
        if self.col_drop:
            X_copy.drop(self.col_drop, axis=1, inplace=True)

        return X_copy


class AddAttributes_Numerical(BaseEstimator, TransformerMixin):
    """
    A transformer to add new numerical attributes for feature engineering in a dataset.

    Parameters:
    -----------
    add_attributes : bool, default=True
        If True, additional attributes are calculated and added to the dataset.

    Features added:
    --------------
    pct_half_bath : float
        Percentage of half bathrooms relative to total bathrooms.
    timing_remodel_index : float
        Timing of remodeling relative to house age (0-1 scale).
    total_area : float
        Total area including all living spaces and outdoor features.
    pct_finished_bsmt_sf : float
        Percentage of basement that is finished.
    """

    def __init__(self, add_attributes: bool = True) -> None:
        self.add_attributes = add_attributes

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "AddAttributes_Numerical":
        """Fit method - no fitting required for feature engineering."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method to add new numerical attributes.

        Parameters:
        -----------
        X : DataFrame
            Input features.

        Returns:
        --------
        X_copy : DataFrame
            DataFrame with added attributes.
        """
        if not self.add_attributes:
            return X

        X_copy = X.copy()  # Avoid modifying the original dataset

        # Safely calculate pct_half_bath
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_half_bath = np.divide(
                0.5 * X_copy["half_bath"] + 0.5 * X_copy["bsmt_half_bath"],
                X_copy["full_bath"]
                + 0.5 * X_copy["half_bath"]
                + X_copy["bsmt_full_bath"]
                + 0.5 * X_copy["bsmt_half_bath"],
                out=np.zeros_like(X_copy["half_bath"], dtype=float),
                where=(
                    X_copy["full_bath"]
                    + 0.5 * X_copy["half_bath"]
                    + X_copy["bsmt_full_bath"]
                    + 0.5 * X_copy["bsmt_half_bath"]
                )
                != 0,
            )

        # Safely calculate timing_remodel_index
        timing_numerator = X_copy["year_remod/add"] - X_copy["year_blt"]
        timing_denominator = X_copy["year_sold"] - X_copy["year_blt"]
        timing_remodel_index = np.divide(
            np.maximum(0, timing_numerator),  # Ensure numerator is non-negative
            timing_denominator,
            out=np.zeros_like(timing_numerator, dtype=float),
            where=(timing_denominator != 0),
        )

        # Calculate total_area
        total_area = (
            X_copy["1st_flr_sf"]
            + X_copy["2nd_flr_sf"]
            + X_copy["gr_liv_area"]
            + X_copy["total_bsmt_sf"]
            + X_copy["pool_area"]
            + X_copy["open_porch_sf"]
            + X_copy["enclosed_porch"]
            + X_copy["3ssn_porch"]
            + X_copy["screen_porch"]
            + X_copy["wood_deck_sf"]
        )

        # Safely calculate pct_finished_bsmt_sf
        pct_finished_bsmt_sf = np.divide(
            X_copy["bsmtfin_sf_1"] + X_copy["bsmtfin_sf_2"],
            X_copy["total_bsmt_sf"],
            out=np.zeros_like(X_copy["bsmtfin_sf_1"], dtype=float),
            where=(X_copy["total_bsmt_sf"] != 0),
        )

        # Add new features
        X_copy["pct_half_bath"] = pct_half_bath
        X_copy["timing_remodel_index"] = timing_remodel_index
        X_copy["total_area"] = total_area
        X_copy["pct_finished_bsmt_sf"] = pct_finished_bsmt_sf

        return X_copy


class AddAttributes_Ordinal(BaseEstimator, TransformerMixin):
    """
    A transformer to add ordinal attributes for feature engineering.

    Parameters:
    -----------
    add_attributes : bool, default=True
        Whether to add new attributes.
    proximity_data : dict, default=None
        Dictionary mapping neighborhoods to university proximity categories.

    Features added:
    --------------
    university_proximity_category : int
        Category indicating proximity to university (-1 if unknown).
    interior_quality_score : float
        Composite score of interior quality features.
    exterior_quality_score : float
        Composite score of exterior quality features.
    """

    def __init__(self, add_attributes=True, proximity_data=None):
        if proximity_data is not None and not isinstance(proximity_data, dict):
            raise ValueError("proximity_data must be a dictionary or None.")

        self.add_attributes = add_attributes
        self.proximity_data = proximity_data

    def fit(self, X, y=None):
        """Fit method - no fitting required for feature engineering."""
        return self

    def get_university_proximity_category(self, neighborhood):
        """
        Helper method to get the university proximity category for a given neighborhood.

        Parameters:
        -----------
        neighborhood : str
            Name of the neighborhood.

        Returns:
        --------
        int
            Proximity category corresponding to the neighborhood (-1 if not found).
        """
        if self.proximity_data and neighborhood in self.proximity_data:
            return self.proximity_data[neighborhood]
        return -1  # Default category if neighborhood is not found

    def transform(self, X):
        """
        Transform method to add ordinal attributes.

        Parameters:
        -----------
        X : DataFrame
            Input features.

        Returns:
        --------
        DataFrame
            DataFrame with added ordinal attributes if add_attributes is True,
            otherwise returns original DataFrame.
        """
        if not self.add_attributes:
            return X

        X_copy = X.copy()

        # Add the university proximity category to the dataset
        X_copy["university_proximity_category"] = X_copy["neighborhood"].apply(
            lambda neighborhood: self.get_university_proximity_category(neighborhood)
        )

        # Convert columns to numeric before adding
        interior_cols = [
            "kitchen_qu",
            "bsmt_qu",
            "bsmt_cond",
            "garage_finish",
            "bsmt_exposure",
            "fireplace_qu",
            "electrical",
            "garage_qu",
            "garage_cond",
            "utilities",
        ]
        exterior_cols = [
            "exter_qu",
            "exter_cond",
            "paved_drive",
            "land_slope",
            "lot_shape",
            "fence",
            "pool_qu",
        ]

        # Calculate interior quality score by converting each column to numeric first
        interior_quality_score = pd.Series(0, index=X_copy.index)
        for col in interior_cols:
            if col in X_copy.columns:
                interior_quality_score += pd.to_numeric(
                    X_copy[col], errors="coerce"
                ).fillna(0)
        X_copy["interior_quality_score"] = interior_quality_score

        # Calculate exterior quality score by converting each column to numeric first
        exterior_quality_score = pd.Series(0, index=X_copy.index)
        for col in exterior_cols:
            if col in X_copy.columns:
                exterior_quality_score += pd.to_numeric(
                    X_copy[col], errors="coerce"
                ).fillna(0)
        X_copy["exterior_quality_score"] = exterior_quality_score

        return X_copy
