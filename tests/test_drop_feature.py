import pandas as pd
import pytest

from iowa_dream.feature_engineering.add_drop_features import DropFeatures


def test_drop_features():
    # Create a sample DataFrame
    data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
    df = pd.DataFrame(data)

    # Initialize the DropFeatures transformer with columns to drop
    transformer = DropFeatures(col_drop=["B", "C"])

    # Transform the DataFrame
    transformed_df = transformer.transform(df)

    # Check if the columns 'B' and 'C' are dropped
    assert "B" not in transformed_df.columns
    assert "C" not in transformed_df.columns
    assert "A" in transformed_df.columns


if __name__ == "__main__":
    pytest.main()
