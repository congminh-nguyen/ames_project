import pytest

from iowa_dream.feature_engineering.categotical_transformer import (
    NominalTransformer,
    OrdinalMerger,
)


# Test NominalTransformer
@pytest.fixture
def nominal_test_data():
    return pd.DataFrame(
        {
            "year_blt": [2000, 1990, 1980, 2010],
            "year_sold": [2024, 2024, 2024, 2024],
            "year_remod/add": [2005, 1990, 1995, 2010],
            "exterior_1st": ["Brick", "Wood", "Brick", "Vinyl"],
            "exterior_2nd": ["Brick", "Wood", "Vinyl", "Vinyl"],
        }
    )


def test_nominal_transformer(nominal_test_data):
    transformer = NominalTransformer()
    transformed = transformer.fit_transform(nominal_test_data)

    # Check transformed data structure
    assert "year_blt" in transformed.columns
    assert "year_remod/add" in transformed.columns
    assert "exterior_2nd" in transformed.columns

    # Check transformations
    assert list(transformed["exterior_2nd"]) == [1, 1, 0, 1]
    assert list(transformed["year_remod/add"]) == [1, 0, 1, 0]
    assert isinstance(transformed["year_blt"].iloc[0], str)


import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {"col1": [0, 1, 1, 2, 2, 2, 3, 3, 3, 3], "col2": [1, 2, 2, 2, 3, 3, 4, 4, 4, 4]}
    )


def test_ordinal_merger_min_obs_3(sample_data):
    transformer = OrdinalMerger(min_obs=3)
    transformer.fit(sample_data)
    transformed = transformer.transform(sample_data)

    expected = pd.DataFrame(
        {"col1": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3], "col2": [2, 2, 2, 2, 4, 4, 4, 4, 4, 4]}
    )
    pd.testing.assert_frame_equal(transformed, expected)


def test_ordinal_merger_min_obs_1(sample_data):
    transformer = OrdinalMerger(min_obs=1)
    transformer.fit(sample_data)
    transformed = transformer.transform(sample_data)

    # Should return original data since all categories meet min_obs=1
    pd.testing.assert_frame_equal(transformed, sample_data)


def test_ordinal_merger_min_obs_10(sample_data):
    transformer = OrdinalMerger(min_obs=10)
    transformer.fit(sample_data)
    transformed = transformer.transform(sample_data)

    expected = pd.DataFrame(
        {"col1": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], "col2": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}
    )
    pd.testing.assert_frame_equal(transformed, expected)
