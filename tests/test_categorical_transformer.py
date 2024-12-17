import pandas as pd
import pytest

from iowa_dream.feature_engineering.categotical_transformer import (
    NominalGrouper,
    OrdinalMerger,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {"col1": [0, 1, 1, 2, 2, 2, 3, 3, 3, 3], "col2": [1, 2, 2, 2, 3, 3, 4, 4, 4, 4]}
    )


@pytest.fixture
def nominal_data():
    return pd.DataFrame(
        {"category": ["A", "A", "B", "B", "B", "C", "D", "E", "F", "G"]}
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


def test_nominal_grouper_min_obs_3(nominal_data):
    transformer = NominalGrouper(min_obs=3)
    transformed = transformer.fit_transform(nominal_data)

    expected = pd.DataFrame(
        {
            "category": [
                "Other",
                "Other",
                "B",
                "B",
                "B",
                "Other",
                "Other",
                "Other",
                "Other",
                "Other",
            ]
        }
    )
    pd.testing.assert_frame_equal(transformed, expected)
