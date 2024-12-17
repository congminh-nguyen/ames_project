import numpy as np
import pandas as pd
import pytest

from iowa_dream.feature_engineering.lot_frontage_imputer import (  # Assume the class is in a file named group_median_imputer.py
    GroupMedianImputer,
)


def test_group_median_imputer():
    # Input data
    data = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
            "subgroup": [1, 1, 2, 1, 1, 2, 1, 2, 2, 2],
            "value": [1, np.nan, 3, 4, np.nan, 6, np.nan, 8, 2, 7],
        }
    )

    # Expected output
    expected_data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "A", "C", "B", "B", "C", "C", "C"],
            "subgroup": [1, 1, 1, 2, 1, 1, 2, 2, 2, 2],
            "value": [1.0, 1.0, 4.0, 3.0, 4.0, 4.0, 6.0, 8.0, 2.0, 7.0],
        }
    )

    # Instantiate the imputer
    imputer = GroupMedianImputer(group_cols=["group", "subgroup"], target_col="value")

    # Transform the data
    transformed_data = imputer.fit_transform(data)

    # Reset indices for comparison
    transformed_data = transformed_data.reset_index(drop=True)
    expected_data = expected_data.reset_index(drop=True)

    # Assert the transformed data matches the expected output
    pd.testing.assert_frame_equal(transformed_data, expected_data)


if __name__ == "__main__":
    pytest.main()
