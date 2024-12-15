from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, col_drop: Optional[List[str]] = None):
        self.col_drop = col_drop

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DropFeatures":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if self.col_drop:
            X_copy.drop(self.col_drop, axis=1, inplace=True)

        return X_copy


class AddAttributesNumerical(BaseEstimator, TransformerMixin):
    def __init__(self, add_attributes=True):
        self.add_attributes = add_attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.add_attributes:
            pct_half_bath = (0.5 * X["half_bath"] + 0.5 * X["bsmt_half_bath"]) / (
                X["full_bath"]
                + 0.5 * X["half_bath"]
                + X["bsmt_full_bath"]
                + 0.5 * X["bsmt_half_bath"]
            )
            timing_numerator = X["year_remod/add"] - X["year_blt"]
            timing_denominator = X["year_sold"] - X["year_blt"]
            timing_remodel_index = np.where(
                (X["year_sold"] > X["year_blt"]) & (timing_denominator != 0),
                timing_numerator / timing_denominator,  # Perform division safely
                0,  # Default value when conditions are not met
            )
            total_area = (
                X["1st_flr_sf"]
                + X["2nd_flr_sf"]
                + X["gr_liv_area"]
                + X["total_bsmt_sf"]
                + X["pool_area"]
                + X["open_porch_sf"]
                + X["enclosed_porch"]
                + X["3ssn_porch"]
                + X["screen_porch"]
                + X["wood_deck_sf"]
            )
            pct_finished_bsmt_sf = (X["bsmtfin_sf_1"] + X["bsmtfin_sf_2"]) / X[
                "total_bsmt_sf"
            ]
            new_features = pd.DataFrame(
                {
                    "pct_half_bath": pct_half_bath,
                    "timing_remodel_index": timing_remodel_index,
                    "total_area": total_area,
                    "pct_finished_bsmt_sf": pct_finished_bsmt_sf,
                },
                index=X.index,
            )
            X_copy = pd.concat([X, new_features], axis=1)
            return X_copy
        else:
            return X
