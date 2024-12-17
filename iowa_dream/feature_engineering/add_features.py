from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Add_Drop_Attributes(BaseEstimator, TransformerMixin):
    """Transformer that combines full_bath and half_bath into total_bath, calculates living area percentage,
    creates kitchen quality score and fireplace score, adds timing_remodel_index, creates recession indicator,
    adds university proximity category, calculates neighborhood score and basement total score.
    """

    def __init__(self, proximity_data: Optional[dict] = None):
        self.proximity_data = proximity_data

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def get_university_proximity_category(self, neighborhood: str) -> int:
        """Helper method to get the university proximity category for a given neighborhood.

        Parameters
        ----------
        neighborhood : str
            Name of the neighborhood.

        Returns
        -------
        int
            Proximity category corresponding to the neighborhood (-1 if not found).
        """
        if self.proximity_data and neighborhood in self.proximity_data:
            return self.proximity_data[neighborhood]
        return -1  # Default category if neighborhood is not found

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        # Calculate total baths (counting half baths as 0.5)
        X_copy["total_bath"] = X_copy["full_bath"] + 0.5 * X_copy["half_bath"]

        # Calculate living area percentage
        total_area = (
            X_copy["lot_area"]
            + X_copy["total_bsmt_sf"]
            + X_copy["gr_liv_area"]
            + X_copy["wood_deck_sf"]
        )
        X_copy["living_area_percentage"] = np.divide(
            X_copy["gr_liv_area"],
            total_area,
            out=np.zeros_like(X_copy["gr_liv_area"], dtype=float),
            where=(total_area != 0),
        )

        # Calculate kitchen quality score
        X_copy["kitchen_quality_score"] = X_copy["kitchen_abvgr"] * X_copy["kitchen_qu"]

        # Calculate fireplace score
        X_copy["fire_place_score"] = X_copy["fireplace_qu"] * X_copy["fireplaces"]

        # Calculate timing remodel index
        timing_numerator = X_copy["year_remod/add"] - X_copy["year_blt"]
        timing_denominator = X_copy["year_sold"] - X_copy["year_blt"]
        X_copy["timing_remodel_index"] = np.divide(
            np.maximum(0, timing_numerator),  # Ensure numerator is non-negative
            timing_denominator,
            out=np.zeros_like(timing_numerator, dtype=float),
            where=(timing_denominator != 0),
        )

        # Create recession indicator (2008 and after = 1, other years = 0)
        X_copy["recession_period"] = (X_copy["year_sold"] >= 2008).astype(int)

        # Add university proximity category
        X_copy["university_proximity_category"] = (
            X_copy["neighborhood"]
            .apply(
                lambda neighborhood: self.get_university_proximity_category(
                    neighborhood
                )
            )
            .astype(int)
        )

        # Calculate neighborhood score
        X_copy["neighborhood_score"] = (
            X_copy["overall_qu"] + X_copy["overall_cond"] + X_copy["exter_cond"]
        )

        # Compute mean neighborhood score for each neighborhood
        neighborhood_mean_score = X_copy.groupby("neighborhood")[
            "neighborhood_score"
        ].transform("mean")
        X_copy["neighborhood_score"] = neighborhood_mean_score

        # Calculate basement total score
        X_copy["bsmt_total_score"] = X_copy["bsmt_qu"] + X_copy["bsmt_cond"]

        # Calculate overall quality and condition score
        X_copy["overall_score"] = X_copy["overall_qu"] + X_copy["overall_cond"]

        # Drop original columns
        X_copy.drop(
            columns=[
                "full_bath",
                "half_bath",
                "kitchen_abvgr",
                "kitchen_qu",
                "fireplace_qu",
                "fireplaces",
                "year_sold",
                "year_remod/add",
                "neighborhood",
                "bsmt_qu",
                "bsmt_cond",
                "overall_qu",
                "overall_cond",
            ],
            inplace=True,
        )

        return X_copy

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X).transform(X)
