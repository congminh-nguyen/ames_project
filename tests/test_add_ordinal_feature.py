import pandas as pd
import pytest

from iowa_dream.feature_engineering.add_drop_features import AddAttributes_Ordinal


@pytest.fixture
def sample_data():
    """Fixture to provide sample DataFrame."""
    data = {
        "neighborhood": ["Downtown", "Suburb", "Rural", "Unknown"],
        "value": [100, 200, 300, 400],
        "kitchen_qu": [1, 2, 3, 4],
        "bsmt_qu": [2, 3, 4, 1],
        "bsmt_cond": [3, 4, 1, 2],
        "garage_finish": [4, 1, 2, 3],
        "bsmt_exposure": [1, 2, 3, 4],
        "fireplace_qu": [2, 3, 4, 1],
        "electrical": [4, 1, 2, 3],
        "garage_qu": [1, 2, 3, 4],
        "garage_cond": [2, 3, 4, 1],
        "utilities": [4, 1, 2, 3],
        "exter_qu": [1, 2, 3, 4],
        "exter_cond": [2, 3, 4, 1],
        "paved_drive": [3, 4, 1, 2],
        "land_slope": [4, 1, 2, 3],
        "lot_shape": [1, 2, 3, 4],
        "fence": [2, 3, 4, 1],
        "pool_qu": [3, 4, 1, 2],
    }
    return pd.DataFrame(data)


@pytest.fixture
def proximity_data():
    """Fixture to provide sample proximity data."""
    return {"Downtown": 1, "Suburb": 2, "Rural": 3}


def test_transform_with_proximity_data(sample_data, proximity_data):
    """Test the transform method when proximity data is provided."""
    transformer = AddAttributes_Ordinal(
        add_attributes=True, proximity_data=proximity_data
    )
    transformed_data = transformer.fit_transform(sample_data)

    # Assert that the new column is added
    assert "university_proximity_category" in transformed_data.columns

    # Assert values are correct
    expected_proximity = [1, 2, 3, -1]
    assert (
        transformed_data["university_proximity_category"].tolist() == expected_proximity
    )

    # Assert quality scores are calculated correctly
    assert "interior_quality_score" in transformed_data.columns
    assert "exterior_quality_score" in transformed_data.columns

    # Expected values based on the sample data
    expected_interior_scores = [
        24,
        22,
        28,
        26,
    ]  # Sum of interior quality features for each row
    expected_exterior_scores = [16, 19, 18, 17]  # Calculated manually for each row

    assert (
        transformed_data["interior_quality_score"].tolist() == expected_interior_scores
    )
    assert (
        transformed_data["exterior_quality_score"].tolist() == expected_exterior_scores
    )


def test_transform_without_proximity_data(sample_data):
    """Test the transform method when proximity data is not provided."""
    transformer = AddAttributes_Ordinal(add_attributes=True, proximity_data=None)
    transformed_data = transformer.fit_transform(sample_data)

    # Assert that the new column is added with default values
    assert "university_proximity_category" in transformed_data.columns
    assert all(transformed_data["university_proximity_category"] == -1)


def test_transform_no_add_attributes(sample_data, proximity_data):
    """Test the transform method when add_attributes is False."""
    transformer = AddAttributes_Ordinal(
        add_attributes=False, proximity_data=proximity_data
    )
    transformed_data = transformer.fit_transform(sample_data)

    # Assert that no new column is added
    assert "university_proximity_category" not in transformed_data.columns
    # Assert original DataFrame is unchanged
    pd.testing.assert_frame_equal(sample_data, transformed_data)


def test_proximity_category_functionality(proximity_data):
    """Test the get_university_proximity_category method."""
    transformer = AddAttributes_Ordinal(proximity_data=proximity_data)

    assert transformer.get_university_proximity_category("Downtown") == 1
    assert transformer.get_university_proximity_category("Suburb") == 2
    assert transformer.get_university_proximity_category("Rural") == 3
    assert transformer.get_university_proximity_category("Unknown") == -1
    assert transformer.get_university_proximity_category(None) == -1


def test_input_validation():
    """Test input validation for incorrect proximity_data."""
    with pytest.raises(ValueError):
        AddAttributes_Ordinal(add_attributes=True, proximity_data="invalid_data")
