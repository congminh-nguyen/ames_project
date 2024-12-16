import pandas as pd
import pytest

from iowa_dream.feature_engineering.add_drop_features import AddAttributesNumerical


# Test function
def test_add_attributes_numerical():
    # Create a sample DataFrame
    data = {
        "half_bath": [1, 0, 2],
        "bsmt_half_bath": [0, 1, 0],
        "full_bath": [2, 1, 2],
        "bsmt_full_bath": [1, 1, 1],
        "year_remod/add": [2005, 1999, 2006],
        "year_blt": [2000, 1995, 2005],
        "year_sold": [2010, 2000, 2005],
        "1st_flr_sf": [1000, 800, 1200],
        "2nd_flr_sf": [500, 400, 600],
        "gr_liv_area": [1500, 1200, 1800],
        "total_bsmt_sf": [0, 600, 1000],
        "pool_area": [0, 0, 0],
        "open_porch_sf": [100, 50, 150],
        "enclosed_porch": [0, 0, 0],
        "3ssn_porch": [0, 0, 0],
        "screen_porch": [0, 0, 0],
        "wood_deck_sf": [200, 150, 250],
        "bsmtfin_sf_1": [0, 300, 500],
        "bsmtfin_sf_2": [0, 100, 300],
    }
    df = pd.DataFrame(data)

    # Initialize the AddAttributesNumerical transformer
    transformer = AddAttributesNumerical(add_attributes=True)

    # Transform the DataFrame
    transformed_df = transformer.transform(df)

    # Check if the new features are added
    assert "pct_half_bath" in transformed_df.columns
    assert "timing_remodel_index" in transformed_df.columns
    assert "total_area" in transformed_df.columns
    assert "pct_finished_bsmt_sf" in transformed_df.columns

    # Expected values for new features
    expected_pct_half_bath = pd.Series([0.142857, 0.200000, 0.250000], index=df.index)
    expected_timing_remodel_index = pd.Series([0.5, 0.8, 0.0], index=df.index)
    expected_total_area = pd.Series([3300, 3200, 5000], index=df.index)
    expected_pct_finished_bsmt_sf = pd.Series([0, 0.666667, 0.8], index=df.index)

    # Test floating-point precision for numerical columns
    pd.testing.assert_series_equal(
        transformed_df["pct_half_bath"],
        expected_pct_half_bath,
        check_names=False,
        atol=1e-6,
    )
    pd.testing.assert_series_equal(
        transformed_df["timing_remodel_index"],
        expected_timing_remodel_index,
        check_names=False,
        atol=1e-6,
    )
    pd.testing.assert_series_equal(
        transformed_df["total_area"], expected_total_area, check_names=False
    )
    pd.testing.assert_series_equal(
        transformed_df["pct_finished_bsmt_sf"],
        expected_pct_finished_bsmt_sf,
        check_names=False,
        atol=1e-6,
    )


if __name__ == "__main__":
    pytest.main()
