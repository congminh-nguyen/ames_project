import numpy as np
import pandas as pd
import pytest

from iowa_dream.feature_engineering.add_features import Add_Drop_Attributes


@pytest.fixture
def sample_input_data():
    return pd.DataFrame(
        {
            "full_bath": [2, 1],
            "half_bath": [1, 0],
            "lot_area": [8000, 9000],
            "total_bsmt_sf": [1000, 800],
            "gr_liv_area": [2000, 1800],
            "wood_deck_sf": [100, 0],
            "kitchen_abvgr": [1, 1],
            "kitchen_qu": [3, 4],
            "fireplace_qu": [4, 0],
            "fireplaces": [1, 0],
            "year_blt": [1980, 1990],
            "year_remod/add": [1995, 1990],
            "year_sold": [2008, 2006],
            "neighborhood": ["OldTown", "NAmes"],
            "overall_qu": [7, 6],
            "overall_cond": [5, 6],
            "exter_cond": [3, 3],
            "bsmt_qu": [4, 3],
            "bsmt_cond": [3, 3],
        }
    )


@pytest.fixture
def proximity_data():
    return {"OldTown": 4, "NAmes": 2}


def test_output_is_dataframe(sample_input_data, proximity_data):
    transformer = Add_Drop_Attributes(proximity_data=proximity_data)
    output = transformer.transform(sample_input_data)
    assert isinstance(output, pd.DataFrame)


def test_expected_columns_exist(sample_input_data, proximity_data):
    transformer = Add_Drop_Attributes(proximity_data=proximity_data)
    output = transformer.transform(sample_input_data)

    expected_columns = [
        "total_bath",
        "living_area_percentage",
        "kitchen_quality_score",
        "fire_place_score",
        "timing_remodel_index",
        "recession_period",
        "university_proximity_category",
        "neighborhood_score",
        "bsmt_total_score",
        "overall_score",
    ]

    for col in expected_columns:
        assert col in output.columns


def test_total_bath_calculation(sample_input_data, proximity_data):
    transformer = Add_Drop_Attributes(proximity_data=proximity_data)
    output = transformer.transform(sample_input_data)
    assert output["total_bath"].iloc[0] == 2.5  # 2 full + 0.5 * 1 half


def test_kitchen_and_fireplace_scores(sample_input_data, proximity_data):
    transformer = Add_Drop_Attributes(proximity_data=proximity_data)
    output = transformer.transform(sample_input_data)

    assert output["kitchen_quality_score"].iloc[0] == 3  # 1 * 3
    assert output["fire_place_score"].iloc[0] == 4  # 4 * 1
    assert output["fire_place_score"].iloc[1] == 0  # 0 * 0


def test_timing_remodel_index(sample_input_data, proximity_data):
    transformer = Add_Drop_Attributes(proximity_data=proximity_data)
    output = transformer.transform(sample_input_data)

    assert (
        output["timing_remodel_index"].iloc[0] == 0.5357142857142857
    )  # (1995-1980)/(2008-1980)
    assert (
        output["timing_remodel_index"].iloc[1] == 0
    )  # (1990-1990)/(2006-1990) = 0/16 = 0


def test_recession_period(sample_input_data, proximity_data):
    transformer = Add_Drop_Attributes(proximity_data=proximity_data)
    output = transformer.transform(sample_input_data)

    assert output["recession_period"].iloc[0] == 1  # 2008 is in recession period
    assert output["recession_period"].iloc[1] == 0  # 2006 is not in recession period


def test_proximity_and_scores(sample_input_data, proximity_data):
    transformer = Add_Drop_Attributes(proximity_data=proximity_data)
    output = transformer.transform(sample_input_data)

    assert output["university_proximity_category"].iloc[0] == 4  # OldTown category
    assert output["university_proximity_category"].iloc[1] == 2  # NAmes category
    assert output["neighborhood_score"].iloc[0] == 15  # Mean of (7+5+3) for OldTown
    assert output["neighborhood_score"].iloc[1] == 15  # Mean of (6+6+3) for NAmes
    assert output["bsmt_total_score"].iloc[0] == 7  # 4 + 3
    assert output["bsmt_total_score"].iloc[1] == 6  # 3 + 3
    assert output["overall_score"].iloc[0] == 12  # 7 + 5
    assert output["overall_score"].iloc[1] == 12  # 6 + 6


def test_living_area_percentage(sample_input_data, proximity_data):
    transformer = Add_Drop_Attributes(proximity_data=proximity_data)
    output = transformer.transform(sample_input_data)

    total_area_0 = 8000 + 1000 + 2000 + 100  # lot + basement + living + deck
    expected_percentage_0 = 2000 / total_area_0
    assert np.isclose(output["living_area_percentage"].iloc[0], expected_percentage_0)

    total_area_1 = 9000 + 800 + 1800 + 0  # lot + basement + living + deck
    expected_percentage_1 = 1800 / total_area_1
    assert np.isclose(output["living_area_percentage"].iloc[1], expected_percentage_1)


def test_columns_are_dropped(sample_input_data, proximity_data):
    transformer = Add_Drop_Attributes(proximity_data=proximity_data)
    output = transformer.transform(sample_input_data)

    dropped_columns = [
        "full_bath",
        "half_bath",
        "kitchen_abvgr",
        "kitchen_qu",
        "fireplace_qu",
        "fireplaces",
        "year_sold",
        "year_remod/add",
        "neighborhood",
        "bsmt_qu",
        "bsmt_cond",
    ]

    for col in dropped_columns:
        assert col not in output.columns
