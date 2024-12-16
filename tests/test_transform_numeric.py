import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from iowa_dream.feature_engineering.numerical_transformer import (
    RobustScalerWithIndicator,
)


@pytest.fixture
def sample_data():
    """Fixture to provide sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],  # No zeros
            "B": [10, 20, 30, 0, 50],  # Has one zero
            "C": [0, 0, 0, 0, 0],  # All zeros
        }
    )


def test_fit(sample_data):
    """Test the fit method stores the correct scaling parameters."""
    scaler = RobustScalerWithIndicator()
    scaler.fit(sample_data)
    assert "A" in scaler.scalers_
    assert "B" in scaler.scalers_
    assert "C" in scaler.scalers_

    assert scaler.scalers_["A"]["center"] == 3
    assert scaler.scalers_["A"]["scale"] == 2
    assert scaler.scalers_["B"]["center"] == 20
    assert scaler.scalers_["B"]["scale"] == 20
    assert scaler.scalers_["C"]["scale"] == 1.0  # Avoid division by zero

    # Test has_zeros_ flags
    assert not scaler.has_zeros_["A"]  # No zeros
    assert scaler.has_zeros_["B"]  # Has zeros
    assert scaler.has_zeros_["C"]  # All zeros


def test_transform(sample_data):
    """Test the transform method applies scaling correctly and adds zero indicators."""
    scaler = RobustScalerWithIndicator()
    scaler.fit(sample_data)
    transformed = scaler.transform(sample_data)

    # Check scaled values for column A
    expected_A = (sample_data["A"] - 3) / 2
    assert np.allclose(transformed["A"], expected_A)
    # No zero indicator for A since it has no zeros
    assert "A_zero_indicator" not in transformed.columns

    # Check scaled values for column B
    expected_B = (sample_data["B"] - 20) / 20
    assert np.allclose(transformed["B"], expected_B)
    # Check zero indicator for column B since it has zeros
    assert np.array_equal(
        transformed["B_zero_indicator"], (sample_data["B"] == 0).astype(int)
    )


def test_inverse_transform(sample_data):
    """Test the inverse_transform method restores original values."""
    scaler = RobustScalerWithIndicator()
    scaler.fit(sample_data)
    transformed = scaler.transform(sample_data)
    inversed = scaler.inverse_transform(transformed)

    # Original values should be restored
    assert np.allclose(inversed["A"], sample_data["A"])
    assert np.allclose(inversed["B"], sample_data["B"])


def test_transform_with_non_dataframe_input():
    """Test the transform method handles numpy arrays correctly."""
    data = np.array([[1, 10], [2, 20], [3, 30], [4, 0], [5, 50]])
    scaler = RobustScalerWithIndicator()
    scaler.fit(data)
    transformed = scaler.transform(data)

    # Check the shape and values
    assert transformed.shape[0] == data.shape[0]
    # Column 0 has no zeros, so no indicator needed
    assert "0_zero_indicator" not in transformed.columns
    # Column 1 has zeros, so indicator needed
    assert "1_zero_indicator" in transformed.columns


def test_inverse_transform_with_zero_indicator():
    """Test inverse_transform skips zero indicator columns."""
    scaler = RobustScalerWithIndicator()
    data = pd.DataFrame({"A": [1, 2, 3, 0, 5]})
    scaler.fit(data)
    transformed = scaler.transform(data)

    # Ensure inverse_transform doesn't process indicator column
    inversed = scaler.inverse_transform(transformed)
    assert "A_zero_indicator" not in inversed.columns


def test_transform_before_fit(sample_data):
    """Test transform raises an error if called before fit."""
    scaler = RobustScalerWithIndicator()
    with pytest.raises(NotFittedError):
        scaler.transform(sample_data)


def test_missing_columns_in_transform(sample_data):
    """Test transform raises an error if input has missing columns."""
    scaler = RobustScalerWithIndicator()
    scaler.fit(sample_data)

    # Remove a column
    modified_data = sample_data.drop(columns=["B"])
    with pytest.raises(ValueError, match="Missing columns"):
        scaler.transform(modified_data)
