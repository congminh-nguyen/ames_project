from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Add_Drop_Attributes(BaseEstimator, TransformerMixin):
    """Transformer that combines full_bath and half_bath into total_bath,
    adds university proximity category, calculates neighborhood score and overall score.
    """

    def __init__(
        self,
        proximity_data: Optional[dict] = None,
        features: Union[str, List[str]] = "all",
        drop_original: bool = True,
    ):
        """Initialize the transformer.

        Parameters
        ----------
        proximity_data : Optional[dict], default=None
            Dictionary mapping neighborhoods to proximity categories
        features : Union[str, List[str]], default='all'
            Features to create. Can be 'all' or a list of feature names from:
            ['total_bath', 'university_proximity_category', 'neighborhood_score',
             'overall_score', 'interior_qu', 'season_indicator', 'age',
             'has_2nd_floor', 'has_wood_deck', 'pct_unf_sf']
        drop_original : bool, default=True
            Whether to drop original columns used to create new features
        """
        self.proximity_data = proximity_data
        self.features = features
        self.drop_original = drop_original
        self.columns_: List[str] = []
        self.feature_list = [
            "total_bath",
            "university_proximity_category",
            "neighborhood_score",
            "overall_score",
            "interior_qu",
            "season_indicator",
            "age",
            "has_2nd_floor",
            "has_wood_deck",
            "pct_unf_sf",
        ]
        super().__init__()

    def get_feature_names_out(self, input_features=None):
        """Get output feature names.

        Parameters
        ----------
        input_features : list of str or None
            Input features.

        Returns
        -------
        list of str
            Output feature names.
        """
        return input_features

    def set_output(self, *, transform=None):
        """Set output container.

        Parameters
        ----------
        transform : str or None
            Container type for output.

        Returns
        -------
        self
            Transformer instance.
        """
        return self

    def fit(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "Add_Drop_Attributes":
        """Fit the transformer.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : Optional[Union[pd.Series, np.ndarray]], default=None
            Target variable (unused)

        Returns
        -------
        self : Add_Drop_Attributes
            Returns self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if isinstance(self.features, str) and self.features != "all":
            raise ValueError("features must be 'all' or a list of feature names")

        if isinstance(self.features, list):
            invalid_features = set(self.features) - set(self.feature_list)
            if invalid_features:
                raise ValueError(f"Invalid features specified: {invalid_features}")

        # All columns needed for transformations
        required_columns = []
        features_to_create = (
            self.feature_list if self.features == "all" else self.features
        )

        if "total_bath" in features_to_create:
            required_columns.extend(["full_bath", "half_bath"])
        if (
            "university_proximity_category" in features_to_create
            or "neighborhood_score" in features_to_create
        ):
            required_columns.append("neighborhood")
        if any(
            f in features_to_create for f in ["neighborhood_score", "overall_score"]
        ):
            required_columns.extend(["overall_qu", "overall_cond"])
        if "interior_qu" in features_to_create:
            required_columns.extend(
                ["heating_qu", "kitchen_qu", "fireplace_qu", "bsmt_qu", "bsmt_exposure"]
            )
        if "season_indicator" in features_to_create:
            required_columns.append("mo_sold")
        if "age" in features_to_create:
            required_columns.extend(["year_blt", "year_sold"])
        if "has_2nd_floor" in features_to_create:
            required_columns.append("2nd_flr_sf")
        if "has_wood_deck" in features_to_create:
            required_columns.append("wood_deck_sf")
        if "pct_unf_sf" in features_to_create:
            required_columns.extend(["bsmt_unf_sf", "total_bsmt_sf"])

        missing_cols = set(required_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.columns_ = required_columns
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
        """Transform the data by adding engineered features and dropping original columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        pd.DataFrame
            Transformed features
        """
        check_is_fitted(self, "columns_")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        X_copy = X.copy()
        features_to_create = (
            self.feature_list if self.features == "all" else self.features
        )
        columns_to_drop = set()

        if "total_bath" in features_to_create:
            X_copy["total_bath"] = X_copy["full_bath"] + 0.5 * X_copy["half_bath"]
            columns_to_drop.update(["full_bath", "half_bath"])

        if "university_proximity_category" in features_to_create:
            X_copy["university_proximity_category"] = (
                X_copy["neighborhood"]
                .apply(
                    lambda neighborhood: self.get_university_proximity_category(
                        neighborhood
                    )
                )
                .astype(int)
            )
            if "neighborhood_score" not in features_to_create:
                columns_to_drop.add("neighborhood")

        if "neighborhood_score" in features_to_create:
            X_copy["neighborhood_score"] = X_copy["overall_qu"] + X_copy["overall_cond"]
            neighborhood_mean_score = X_copy.groupby("neighborhood")[
                "neighborhood_score"
            ].transform("mean")
            X_copy["neighborhood_score"] = neighborhood_mean_score
            if "overall_score" not in features_to_create:
                columns_to_drop.update(["overall_qu", "overall_cond"])
            columns_to_drop.add("neighborhood")

        if "overall_score" in features_to_create:
            X_copy["overall_score"] = X_copy["overall_qu"] + X_copy["overall_cond"]
            columns_to_drop.update(["overall_qu", "overall_cond"])

        if "interior_qu" in features_to_create:
            X_copy["interior_qu"] = (
                X_copy["heating_qu"]
                + X_copy["kitchen_qu"]
                + X_copy["fireplace_qu"]
                + X_copy["bsmt_qu"]
                + X_copy["bsmt_exposure"]
            )
            columns_to_drop.update(
                ["heating_qu", "kitchen_qu", "fireplace_qu", "bsmt_qu", "bsmt_exposure"]
            )

        if "season_indicator" in features_to_create:
            X_copy["season_indicator"] = pd.cut(
                X_copy["mo_sold"],
                bins=[0, 2, 5, 8, 12],
                labels=["Winter", "Spring", "Summer", "Fall"],
                include_lowest=True,
            )
            columns_to_drop.add("mo_sold")

        if "age" in features_to_create:
            X_copy["age"] = X_copy["year_sold"] - X_copy["year_blt"]
            columns_to_drop.update(["year_blt", "year_sold"])

        if "has_2nd_floor" in features_to_create:
            X_copy["has_2nd_floor"] = (X_copy["2nd_flr_sf"] > 0).astype(int)
            columns_to_drop.add("2nd_flr_sf")

        if "has_wood_deck" in features_to_create:
            X_copy["has_wood_deck"] = (X_copy["wood_deck_sf"] > 0).astype(int)
            columns_to_drop.add("wood_deck_sf")

        if "pct_unf_sf" in features_to_create:
            X_copy["pct_unf_sf"] = np.where(
                X_copy["total_bsmt_sf"] > 0,
                X_copy["bsmt_unf_sf"] / X_copy["total_bsmt_sf"],
                0,
            )
            columns_to_drop.update(["bsmt_unf_sf"])

        if self.drop_original:
            # Keep all columns except those used to create the requested features
            all_columns = set(X_copy.columns)
            X_copy = X_copy[
                list((all_columns - columns_to_drop) | set(features_to_create))
            ]

        return X_copy

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> pd.DataFrame:
        """Fit and transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : Optional[Union[pd.Series, np.ndarray]], default=None
            Target variable (unused)

        Returns
        -------
        pd.DataFrame
            Transformed features
        """
        return self.fit(X, y).transform(X)
