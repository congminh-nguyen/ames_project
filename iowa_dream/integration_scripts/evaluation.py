# %% Imports
# System imports
import importlib
import sys
from pathlib import Path

# Add the root directory (AMES_PROJECT) to sys.path for module resolution
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Project modules
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

# Reload modules
for module in MODULES:
    if module in sys.modules:
        importlib.reload(sys.modules[module])
    else:
        __import__(module)

import dalex as dx

# Third-party imports
import pandas as pd
from glum import GeneralizedLinearRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

# Project-specific imports
from iowa_dream.data.importer import load_config
from iowa_dream.evaluation.metrics_plot import (
    plot_predictions_and_residuals,
    reevaluate_models,
)
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

# %% Load and Prepare Data
# Example usage
data_file = (
    project_root
    / load_config()["kaggle"]["cleaned_path"]
    / "cleaned_AmesHousing.parquet"
)
df = pd.read_parquet(data_file)

# Get data dictionary from config
config = load_config()
proximity_data = {
    neighborhood: group["category"]
    for group in config["university_proximity"]
    for neighborhood in group["neighborhoods"]
}
glm_data_dict = config["glm_data_dict"]
glm_ordinal_features = glm_data_dict["categorical"]["ordinal"]["columns"]
glm_nominal_features = glm_data_dict["categorical"]["nominal"]["columns"]
glm_numerical_features = glm_data_dict["numerical"]["columns"]
all_features = glm_ordinal_features + glm_nominal_features + glm_numerical_features

# Create train and test splits
df = create_sample_split(df, "pid")
train_df = df[df["sample"] == "train"]
test_df = df[df["sample"] == "test"]
y = df["saleprice"]

# Separate features (X) and target (y)
X_train = train_df.drop(["saleprice", "sample", "pid"], axis=1)
y_train = train_df["saleprice"]
X_test = test_df.drop(["saleprice", "sample", "pid"], axis=1)
y_test = test_df["saleprice"]

# %% Feature Engineering
# Apply imputers and feature transformations
imputer = LotFrontageGroupMedianImputer(
    group_cols=["neighborhood", "lot_config"], target_col="lot_frontage"
)
feature_add_drop = Add_Drop_Attributes(proximity_data=proximity_data)

X_train = imputer.fit_transform(X_train)
X_train = feature_add_drop.fit_transform(X_train)

X_test = imputer.transform(X_test)
X_test = feature_add_drop.transform(X_test)

# %% Define and Train Models
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

# Define GLM pipeline
glm_tuned_pipeline = Pipeline(
    steps=[
        ("preprocessor", glm_with_interaction_preprocessor),
        (
            "glm",
            GeneralizedLinearRegressor(
                family=config["models"]["glm_tuned"]["params"]["family"],
                link=config["models"]["glm_tuned"]["params"]["link"],
                fit_intercept=config["models"]["glm_tuned"]["params"]["fit_intercept"],
                alphas=config["models"]["glm_tuned"]["params"]["alphas"],
                l1_ratio=config["models"]["glm_tuned"]["params"]["l1_ratio"],
                max_iter=config["models"]["glm_tuned"]["params"]["max_iter"],
            ),
        ),
    ]
)

# Train GLM model
glm_tuned_pipeline.fit(X_train, y_train)

# Define LGBM pipeline
LGBM_preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            glm_nominal_features,
        )
    ],
    remainder="passthrough",
)
lgbm_tuned_pipeline = Pipeline(
    [
        ("preprocessor", LGBM_preprocessor),
        (
            "estimator",
            LGBMRegressor(
                objective=config["models"]["lgbm_tuned"]["params"]["objective"],
                n_estimators=config["models"]["lgbm_tuned"]["params"]["n_estimators"],
                learning_rate=config["models"]["lgbm_tuned"]["params"]["learning_rate"],
                num_leaves=config["models"]["lgbm_tuned"]["params"]["num_leaves"],
                min_child_weight=config["models"]["lgbm_tuned"]["params"][
                    "min_child_weight"
                ],
            ),
        ),
    ]
)

# Train LGBM model
lgbm_tuned_pipeline.fit(X_train, y_train)

# %% Model Evaluation
# Reevaluate models on test data
reevaluate_models([glm_tuned_pipeline, lgbm_tuned_pipeline], X_test, y_test)

# Plot predictions and residuals
models = [glm_tuned_pipeline, lgbm_tuned_pipeline]
model_names = ["GLM", "LGBM"]
plot = plot_predictions_and_residuals(
    models=models,
    X_test=X_test,
    y_test=y_test,
    model_names=model_names,
    figsize=(20, 10),
)
plot.show()

# %% Model Explanation
# Create model explainers
glm_tuned_pipeline_exp = dx.Explainer(
    glm_tuned_pipeline,
    X_train,
    y_train,
    label="Tuned GLM with Interaction Terms",
    verbose=0,
)
lgbm_tuned_pipeline_exp = dx.Explainer(
    lgbm_tuned_pipeline, X_train, y_train, label="Tuned LGBM", verbose=0
)

# Evaluate model performance
performance_glm = glm_tuned_pipeline_exp.model_performance()
performance_glm.plot()  # Overall performance metrics

performance_lgbm = lgbm_tuned_pipeline_exp.model_performance()
performance_lgbm.plot()  # Overall performance metrics

# Feature importance
feature_importance_glm = glm_tuned_pipeline_exp.model_parts()
feature_importance_glm.plot(show=True)
feature_importance_lgbm = lgbm_tuned_pipeline_exp.model_parts()
feature_importance_lgbm.plot(show=True)

# %% Partial Dependence and ALE Plots
# Define important features
numerical_important_features = ["gr_liv_area", "age", "total_bsmt_sf"]
categorical_important_features = ["interior_qu", "overall_score"]

# Partial dependence plots for LGBM
pd_lgbm_tuned_pipeline_exp_categorical = lgbm_tuned_pipeline_exp.model_profile(
    variables=categorical_important_features, variable_type="categorical"
)
pd_lgbm_tuned_pipeline_exp_categorical.plot()
pd_lgbm_tuned_pipeline_exp_numerical = lgbm_tuned_pipeline_exp.model_profile(
    variables=numerical_important_features, variable_type="numerical"
)
pd_lgbm_tuned_pipeline_exp_numerical.plot()

# Partial dependence plots for GLM
pd_glm_tuned_pipeline_exp_numerical = glm_tuned_pipeline_exp.model_profile(
    variables=numerical_important_features, variable_type="numerical"
)
pd_glm_tuned_pipeline_exp_numerical.plot()

# ALE plots for LGBM
ale_lgbm_tuned_pipeline_exp_numerical = lgbm_tuned_pipeline_exp.model_profile(
    variables=numerical_important_features, variable_type="numerical", type="ale"
)
ale_lgbm_tuned_pipeline_exp_numerical.plot()

# %% Individual Predictions
# Breakdown plot for a single prediction
breakdown = lgbm_tuned_pipeline_exp.predict_parts(X_train.iloc[0])
breakdown.plot(show=True)

# SHAP plot for a single prediction
shap = lgbm_tuned_pipeline_exp.predict_parts(X_train.iloc[0], type="shap")
shap.plot(show=True)

# Ceteris Paribus profile for a single prediction
cp_profile = lgbm_tuned_pipeline_exp.predict_profile(X_train.iloc[0])
cp_profile.plot()
# %%
