# %% Imports
# System imports
import importlib
import sys
from pathlib import Path

# Add the root directory (AMES_PROJECT) to sys.path for module resolution
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Third-party imports
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
from iowa_dream.feature_engineering.lot_frontage_imputer import (
    LotFrontageGroupMedianImputer,
)
from iowa_dream.models.custom_obj_lgbm import custom_regression_loss
from iowa_dream.models.optuna_objective import optuna_objective
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


# %% Baseline Pipeline
# Create a preprocessor that handles both numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            nominal_features,
        )
    ],
    remainder="passthrough",
)

# Create the full pipeline with imputer, preprocessing and model
lgbm_baseline_pipeline = Pipeline(
    [
        (
            "imputer",
            LotFrontageGroupMedianImputer(
                group_cols=["neighborhood", "lot_config"], target_col="lot_frontage"
            ),
        ),
        ("preprocessor", preprocessor),
        ("estimator", LGBMRegressor(objective="gamma")),
    ]
)

# Fit the baseline pipeline
lgbm_baseline_pipeline.fit(X_train, y_train)

# %% Pipeline with Additional Features
# Create a preprocessor that handles both numerical and categorical features
combined_feature_preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            glm_nominal_features,
        )
    ],
    remainder="passthrough",
)

# Create the full pipeline with imputer, preprocessing and model
lgbm_1_pipeline = Pipeline(
    [
        (
            "imputer",
            LotFrontageGroupMedianImputer(
                group_cols=["neighborhood", "lot_config"], target_col="lot_frontage"
            ),
        ),
        ("feature_add_drop", Add_Drop_Attributes(proximity_data=proximity_data)),
        ("preprocessor", combined_feature_preprocessor),
        ("estimator", LGBMRegressor(objective="gamma")),
    ]
)

# Fit the pipeline with additional features
lgbm_1_pipeline.fit(X_train, y_train)

# %% Optuna Study for Hyperparameter Tuning
# Best Parameters: {'n_estimators': 350, 'learning_rate': 0.03258641681250722, 'num_leaves': 30, 'min_child_weight': 1.3675858461417085}
# Create an Optuna study
study = optuna.create_study(direction="minimize", study_name="Pipeline Optimization")

# Optimize using the defined objective function
study.optimize(
    lambda trial: optuna_objective(trial, X_train, y_train, lgbm_1_pipeline),
    n_trials=20,  # Number of trials (adjust based on time/needs)
    show_progress_bar=True,
)

# Retrieve the best parameters
best_params = study.best_params
print("Best Parameters:", best_params)

# %% Tuned Pipeline with Best Parameters
lgbm_tuned_pipeline = Pipeline(
    [
        (
            "imputer",
            LotFrontageGroupMedianImputer(
                group_cols=["neighborhood", "lot_config"], target_col="lot_frontage"
            ),
        ),
        ("feature_add_drop", Add_Drop_Attributes(proximity_data=proximity_data)),
        ("preprocessor", combined_feature_preprocessor),
        (
            "estimator",
            LGBMRegressor(
                objective="regression",
                n_estimators=350,
                learning_rate=0.03,
                num_leaves=30,
                min_child_weight=1.37,
            ),
        ),
    ]
)

# Fit the pipeline with the best parameters
lgbm_tuned_pipeline.fit(X_train, y_train)

# %% Pipeline with Custom Loss Function
# Modify the pipeline to use the custom objective directly
lgbm_custom_loss_pipeline = Pipeline(
    [
        (
            "imputer",
            LotFrontageGroupMedianImputer(
                group_cols=["neighborhood", "lot_config"], target_col="lot_frontage"
            ),
        ),
        ("feature_add_drop", Add_Drop_Attributes(proximity_data=proximity_data)),
        ("preprocessor", combined_feature_preprocessor),
        (
            "estimator",
            LGBMRegressor(objective=custom_regression_loss),
        ),  # Remove fixed n_estimators for tuning
    ]
)

# %% Grid Search for Custom Loss Pipeline
# Define the parameter grid
param_grid = {
    "estimator__n_estimators": [100, 300],
    "estimator__learning_rate": [0.01, 0.05],
    "estimator__num_leaves": [150, 300],
    "estimator__min_child_weight": [1.0, 5.0],
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    lgbm_custom_loss_pipeline,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5,
    verbose=0,
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best Parameters:", grid_search.best_params_)
# Best Parameters: {'estimator__learning_rate': 0.05,
# 'estimator__min_child_weight': 1.0,
# 'estimator__n_estimators': 300,
# 'estimator__num_leaves': 150}

# %% Tuned Pipeline with Custom Loss Function
# Modify the pipeline to use the custom objective directly
lgbm_custom_loss_tuned_pipeline = Pipeline(
    [
        (
            "imputer",
            LotFrontageGroupMedianImputer(
                group_cols=["neighborhood", "lot_config"], target_col="lot_frontage"
            ),
        ),
        ("feature_add_drop", Add_Drop_Attributes(proximity_data=proximity_data)),
        ("preprocessor", combined_feature_preprocessor),
        (
            "estimator",
            LGBMRegressor(
                objective=custom_regression_loss,
                learning_rate=0.05,
                min_child_weight=1.0,
                n_estimators=300,
                num_leaves=150,
            ),
        ),  # Add best parameters from grid search
    ]
)

# Fit the tuned pipeline with custom loss function
lgbm_custom_loss_tuned_pipeline.fit(X_train, y_train)

# %% Reevaluate Models
# Reevaluate the models using the test set
reevaluate_models(
    [lgbm_tuned_pipeline, lgbm_custom_loss_tuned_pipeline], X_test, y_test
)

# %%
