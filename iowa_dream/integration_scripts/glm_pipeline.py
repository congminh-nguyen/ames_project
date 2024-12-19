# %% Imports
# System imports
import importlib
import sys
from pathlib import Path

# Add the root directory (AMES_PROJECT) to sys.path for module resolution
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# Third-party imports
import pandas as pd
from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

# %% Project modules
# List of project-specific modules to be reloaded
MODULES = [
    "iowa_dream.utils",
    "iowa_dream.data",
    "iowa_dream.feature_engineering",
    "iowa_dream.feature_engineering.lot_frontage_imputer",
    "iowa_dream.evaluation",
    "iowa_dream.evaluation.metrics_plot",
    "iowa_dream.models.optuna_objective",
    "iowa_dream.models.custom_obj_lgbm",
]

# %% Reload modules
# Reload modules if they are already imported, otherwise import them
for module in MODULES:
    if module in sys.modules:
        importlib.reload(sys.modules[module])
    else:
        __import__(module)

# %% Project imports
# Import project-specific functions and classes
from iowa_dream.data.importer import load_config
from iowa_dream.evaluation.metrics_plot import reevaluate_models
from iowa_dream.feature_engineering.add_drop_features import Add_Drop_Attributes
from iowa_dream.feature_engineering.categotical_transformer import (
    NominalGrouper,
    OrdinalMerger,
)
from iowa_dream.feature_engineering.lot_frontage_imputer import (
    LotFrontageGroupMedianImputer,
)
from iowa_dream.feature_engineering.numerical_transformer import WinsorizedRobustScaler
from iowa_dream.utils.sample_split import create_sample_split

# %% Load data
# Load cleaned data from parquet file
data_file = (
    Path(project_root)
    / load_config()["kaggle"]["cleaned_path"]
    / "cleaned_AmesHousing.parquet"
)
df = pd.read_parquet(data_file)

# %%
# Split the data into training and testing sets based on 'pid'
df = create_sample_split(df, "pid")
train_df = df[df["sample"] == "train"]
test_df = df[df["sample"] == "test"]
y = df["saleprice"]  # Target variable

# %%
# Get data dictionary from config
config = load_config()
cleaned_data_dict = config["cleaned_data_dict"]

# Extract feature groups from the configuration
ordinal_features = cleaned_data_dict["ordinal"]["columns"]
nominal_features = cleaned_data_dict["nominal"]["columns"]
discrete_features = cleaned_data_dict["discrete"]["columns"]
continuous_features = cleaned_data_dict["continuous"]["columns"]
numeric_features = continuous_features + discrete_features

# Create a mapping of neighborhoods to their proximity categories
proximity_data = {
    neighborhood: group["category"]
    for group in config["university_proximity"]
    for neighborhood in group["neighborhoods"]
}

# Extract GLM-specific feature groups
glm_data_dict = config["glm_data_dict"]
glm_ordinal_features = glm_data_dict["categorical"]["ordinal"]["columns"]
glm_nominal_features = glm_data_dict["categorical"]["nominal"]["columns"]
glm_numerical_features = glm_data_dict["numerical"]["columns"]

# Separate features (X) and target (y) for training and testing
X_train = train_df.drop(["saleprice", "sample", "pid"], axis=1)
y_train = train_df["saleprice"]
X_test = test_df.drop(["saleprice", "sample", "pid"], axis=1)
y_test = test_df["saleprice"]

# %%
# Define a preprocessing pipeline
baseline_preprocessor = ColumnTransformer(
    transformers=[
        (
            "group_impute",
            LotFrontageGroupMedianImputer(
                group_cols=["neighborhood", "lot_config"], target_col="lot_frontage"
            ),
            ["neighborhood", "lot_config", "lot_frontage"],
        ),
        (
            "cat",
            OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"),
            nominal_features,
        ),
    ]
)
baseline_preprocessor.set_output(transform="pandas")

# Define a GLM model pipeline
baseline_GLM_model_pipeline = Pipeline(
    [
        ("preprocess", baseline_preprocessor),
        (
            "estimate",
            GeneralizedLinearRegressor(family="gamma", l1_ratio=1, fit_intercept=True),
        ),
    ]
)

# Fit the GLM model pipeline on the training data
baseline_GLM_model_pipeline.fit(X_train, y_train)

# %%
# Preprocessing pipeline for numerical features
numerical_pipeline = Pipeline(
    steps=[("winsorized_scaler", WinsorizedRobustScaler(range_min=10, range_max=99))]
)

# Preprocessing pipeline for ordinal features
ordinal_pipeline = Pipeline(steps=[("ordinal_merger", OrdinalMerger(min_obs=10))])

# Preprocessing pipeline for nominal features
nominal_pipeline = Pipeline(
    steps=[
        ("nominal_grouper", NominalGrouper(min_obs=10)),
        (
            "onehot",
            OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"),
        ),
    ]
)

# Combine preprocessing pipelines
glm_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, glm_numerical_features),
        ("ord", ordinal_pipeline, glm_ordinal_features),
        ("nom", nominal_pipeline, glm_nominal_features),
    ]
)

# Full pipeline
glm_1_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            LotFrontageGroupMedianImputer(
                group_cols=["neighborhood", "lot_config"], target_col="lot_frontage"
            ),
        ),
        ("feature_add_drop", Add_Drop_Attributes(proximity_data=proximity_data)),
        ("preprocessor", glm_preprocessor),
        (
            "glm",
            GeneralizedLinearRegressor(
                family="gamma", link="log", l1_ratio=1, fit_intercept=True
            ),
        ),  # GeneralizedLinearRegressor equivalent for gamma family
    ]
)

glm_1_pipeline

# %%
# Fit and evaluate
glm_1_pipeline.fit(X_train, y_train)
# %%
# Define interaction terms
interaction_features = [
    ("age", "exter_qu"),
    ("gr_liv_area", "overall_score"),
    ("gr_liv_area", "neighborhood_score"),
    ("gr_liv_area", "age"),
]

# Preprocessing pipeline for interaction terms
interaction_pipeline = Pipeline(
    steps=[
        (
            "interaction",
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        ),
        (
            "winsorized_scaler",
            WinsorizedRobustScaler(range_min=10, range_max=99),
        ),  # Use WinsorizedRobustScaler to standardize
    ]
)

# Combine preprocessing pipelines
glm_with_interaction_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, glm_numerical_features),
        ("ord", ordinal_pipeline, glm_ordinal_features),
        ("nom", nominal_pipeline, glm_nominal_features),
        (
            "interaction",
            interaction_pipeline,
            [f[0] for f in interaction_features] + [f[1] for f in interaction_features],
        ),
    ]
)

# Full pipeline
glm_tuned_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            LotFrontageGroupMedianImputer(
                group_cols=["neighborhood", "lot_config"], target_col="lot_frontage"
            ),
        ),
        ("feature_add_drop", Add_Drop_Attributes(proximity_data=proximity_data)),
        ("preprocessor", glm_with_interaction_preprocessor),
        (
            "glm",
            GeneralizedLinearRegressorCV(
                family="gamma",
                link="log",
                fit_intercept=True,
                alphas=None,  # default
                min_alpha=None,  # default
                min_alpha_ratio=None,  # default
                l1_ratio=[0, 0.25, 0.5, 0.75, 1.0],
                max_iter=150,
                cv=5,
            ),
        ),
    ]
)

glm_tuned_pipeline
glm_tuned_pipeline.fit(X_train, y_train)
# %%
reevaluate_models(
    [baseline_GLM_model_pipeline, glm_1_pipeline, glm_tuned_pipeline], X_test, y_test
)
