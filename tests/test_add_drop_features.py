import numpy as np
import pandas as pd
import pytest

from iowa_dream.feature_engineering.add_drop_features import Add_Drop_Attributes


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "full_bath": [2, 1],
            "half_bath": [1, 0],
            "neighborhood": ["OldTown", "NAmes"],
            "overall_qu": [7, 6],
            "overall_cond": [5, 6],
            "heating_qu": [3, 4],
            "kitchen_qu": [3, 4],
            "fireplace_qu": [4, 3],
            "bsmt_qu": [4, 3],
            "bsmt_exposure": [2, 3],
            "mo_sold": [3, 7],
            "year_blt": [1980, 1990],
            "year_sold": [2008, 2006],
            "2nd_flr_sf": [800, 0],
            "wood_deck_sf": [100, 0],
            "bsmt_unf_sf": [300, 200],
            "total_bsmt_sf": [1000, 800],
        }
    )


@pytest.fixture
def proximity_data():
    return {"OldTown": 4, "NAmes": 2}


def test_add_drop_attributes_all_features(sample_data, proximity_data):
    # Test with all features
    transformer = Add_Drop_Attributes(
        proximity_data=proximity_data, drop_original=False
    )
    output = transformer.fit_transform(sample_data)

    # Basic feature presence checks
    assert "total_bath" in output.columns
    assert "total_bsmt_sf" in output.columns

    # Basic value checks
    assert output["total_bath"].iloc[0] == 2.5  # 2 + 0.5*1
    assert output["total_bsmt_sf"].iloc[0] == 1000


def test_add_drop_attributes_subset_features(sample_data, proximity_data):
    # Test with subset of features
    selected_features = ["total_bath", "age"]
    transformer = Add_Drop_Attributes(
        proximity_data=proximity_data, features=selected_features, drop_original=False
    )
    output = transformer.fit_transform(sample_data)

    # Check features are present
    assert "total_bath" in output.columns
    assert "age" in output.columns
    assert "total_bsmt_sf" in output.columns

    # Verify calculations
    assert output["total_bath"].iloc[0] == 2.5  # 2 + 0.5*1
    assert output["age"].iloc[0] == 28  # 2008 - 1980


def test_add_drop_attributes_validation(sample_data):
    # Test input validation
    transformer = Add_Drop_Attributes(features="invalid")
    with pytest.raises(
        ValueError, match="features must be 'all' or a list of feature names"
    ):
        transformer.fit(sample_data)

    transformer = Add_Drop_Attributes(features=["invalid_feature"])
    with pytest.raises(ValueError, match="Invalid features specified:"):
        transformer.fit(sample_data)

    transformer = Add_Drop_Attributes()
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame."):
        transformer.fit(np.array([1, 2, 3]))
