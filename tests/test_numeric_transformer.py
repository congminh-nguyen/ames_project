import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from iowa_dream.feature_engineering.numerical_transformer import NumericalTransformer


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [0, 2, 4, 6, 8],
            "feature_3": [10, 20, 30, 40, 50],
        }
    )


def test_numerical_transformer(sample_data):
    """Test the NumericalTransformer."""
    transformer = NumericalTransformer()
    transformed = transformer.fit_transform(sample_data)

    # Check output is a DataFrame with same shape and columns
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape == sample_data.shape
    assert all(col in transformed.columns for col in sample_data.columns)
    assert transformed.index.equals(sample_data.index)

    # Check values are standardized (mean ~0, std ~1) after both power transform and scaling
    for col in transformed.columns:
        assert abs(transformed[col].mean()) < 0.0001
        assert (
            abs(transformed[col].std() - 1.0) < 0.2
        )  # Increased tolerance since we do Yeo-Johnson then scale


def test_not_fitted_error():
    """Test that NotFittedError is raised when transform is called before fit."""
    transformer = NumericalTransformer()
    with pytest.raises(NotFittedError):
        transformer.transform(pd.DataFrame({"col": [1, 2, 3]}))


def test_input_validation():
    """Test that ValueError is raised for non-DataFrame inputs."""
    transformer = NumericalTransformer()
    with pytest.raises(ValueError):
        transformer.fit(np.array([[1, 2], [3, 4]]))


def test_missing_columns():
    """Test that KeyError is raised when columns are missing during transform."""
    transformer = NumericalTransformer()
    train_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    transformer.fit(train_df)

    test_df = pd.DataFrame({"col1": [1, 2]})  # Missing col2
    with pytest.raises(KeyError):
        transformer.transform(test_df)


def test_transformation_order():
    """Test that Yeo-Johnson is applied before StandardScaler."""
    transformer = NumericalTransformer()
    data = pd.DataFrame({"skewed": [1, 1, 1, 1, 10]})  # Highly skewed data

    transformer.fit(data)

    # Get intermediate Yeo-Johnson transformed data
    yeo_johnson_data = transformer.power_transformer_.transform(data)

    # Then apply standard scaling
    final_transformed = transformer.transform(data)

    # Verify Yeo-Johnson reduced skewness before scaling
    original_skew = abs(data["skewed"].skew())
    yeo_johnson_skew = abs(
        pd.DataFrame(yeo_johnson_data, columns=["skewed"])["skewed"].skew()
    )
    assert yeo_johnson_skew < original_skew

    # Verify final data is standardized
    assert abs(final_transformed["skewed"].mean()) < 0.0001
    assert abs(final_transformed["skewed"].std() - 1.0) < 0.2
