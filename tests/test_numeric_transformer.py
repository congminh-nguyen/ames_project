import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from iowa_dream.feature_engineering.numerical_transformer import WinsorizedRobustScaler


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, 100, -100],  # Contains outliers
            "col2": [10, 20, 30, 40, 50, 60, 70],  # No outliers
        }
    )


def test_winsorized_robust_scaler_direct():
    # Create sample data with outliers
    X = pd.DataFrame(
        {"feature": [-100, 1, 2, 3, 4, 5, 100]}  # Contains outliers at -100 and 100
    )

    # Initialize and fit transformer
    transformer = WinsorizedRobustScaler(range_min=10, range_max=90)
    transformed = transformer.fit_transform(X)

    # The outliers should be clipped and then scaled
    assert transformed[0, 0] < 0  # -100 should be clipped and scaled negative
    assert transformed[-1, 0] > 0  # 100 should be clipped and scaled positive
    assert transformed.shape == X.shape


def test_winsorized_robust_scaler_with_column_transformer(sample_data):
    # Create column transformer that only transforms col1
    column_transformer = ColumnTransformer(
        transformers=[
            (
                "winsorized",
                WinsorizedRobustScaler(range_min=10, range_max=90),
                ["col1"],
            ),
            ("passthrough", "passthrough", ["col2"]),
        ]
    )

    # Fit and transform
    transformed = column_transformer.fit_transform(sample_data)
    transformed_df = pd.DataFrame(transformed, columns=["col1", "col2"])

    # Check that col1 was transformed (outliers handled)
    assert abs(transformed_df["col1"].min()) < abs(sample_data["col1"].min())
    assert abs(transformed_df["col1"].max()) < abs(sample_data["col1"].max())

    # Check that col2 was passed through unchanged
    pd.testing.assert_series_equal(
        transformed_df["col2"],
        sample_data["col2"],
        check_dtype=False,  # Column transformer may change dtype
    )


def test_winsorized_robust_scaler_invalid_range():
    with pytest.raises(
        ValueError,
        match="range_min must be < range_max, both between 0 and 100.",
    ):
        WinsorizedRobustScaler(range_min=101)
