from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline


def reevaluate_models(
    models, X, y, model_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Evaluate multiple models on a test set and return formatted results.

    Parameters
    ----------
    models : list
        List of fitted model objects with predict method
    X : array-like
        Test features
    y : array-like
        True target values
    model_names : list, optional
        List of names for the models. If None, will use generic names

    Returns
    -------
    pd.DataFrame
        DataFrame containing evaluation metrics for each model
    """
    results: Dict[str, list] = {
        "Model": [],
        "RMSE": [],
        "RMSED": [],  # Root Mean Squared Error Degradation
        "MAPE": [],
        "MedAE": [],
        "R-squared": [],
        "Bias": [],
        "Deviance": [],
    }

    # Generate generic model names if none provided
    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(models))]

    # Ensure number of names matches number of models
    if len(model_names) != len(models):
        raise ValueError("Number of model names must match number of models")

    for model, name in zip(models, model_names):
        y_pred = model.predict(X)

        results["Model"].append(name)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        results["RMSE"].append(rmse)
        results["RMSED"].append(rmse / np.mean(y))  # Normalized RMSE
        results["MAPE"].append(mean_absolute_percentage_error(y, y_pred))
        results["MedAE"].append(median_absolute_error(y, y_pred))
        results["R-squared"].append(r2_score(y, y_pred))
        bias = np.mean(y_pred - y)
        deviance = np.mean((y_pred - y) ** 2)
        results["Bias"].append(bias)
        results["Deviance"].append(deviance)

    results_df = pd.DataFrame(results)
    results_df.set_index("Model", inplace=True)

    # Format numeric columns
    for col in results_df.columns:
        if col in ["RMSE", "MedAE", "R-squared", "Bias", "Deviance"]:
            results_df[col] = results_df[col].map("{:.2f}".format)
        else:
            results_df[col] = results_df[col].map("{:.2%}".format)

    return results_df


def analyze_glm_coefficients(
    pipeline: Pipeline,
    numerical_features: List[str],
    ordinal_features: List[str],
    nominal_features: List[str],
    top_n: int = 50,
    pd_options: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Analyze and display GLM coefficients from a fitted pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline containing GLM as final step
    numerical_features : List[str]
        List of numerical feature names
    ordinal_features : List[str]
        List of ordinal feature names
    nominal_features : List[str]
        List of nominal feature names
    top_n : int, default=50
        Number of top features to display
    pd_options : Dict[str, Any], default=None
        Pandas display options to set

    Returns
    -------
    pd.DataFrame
        DataFrame containing feature importance analysis
    """
    # Set pandas display options
    default_options = {
        "display.max_rows": 50,
        "display.max_columns": None,
        "display.width": None,
        "display.float_format": lambda x: f"{x:.4f}",
    }
    pd_options = pd_options or default_options
    for option, value in pd_options.items():
        pd.set_option(option, value)

    # Extract GLM model
    glm_model = pipeline.named_steps["glm"]

    # Get feature names
    nominal_features_encoded = (
        pipeline.named_steps["preprocessor"]
        .transformers_[2][1]
        .named_steps["onehot"]
        .get_feature_names_out(nominal_features)
    )
    interaction_features = (
        pipeline.named_steps["preprocessor"]
        .transformers_[3][1]
        .named_steps["interaction"]
        .get_feature_names_out()
    )

    # Combine all feature names
    all_feature_names = np.concatenate(
        [
            numerical_features,
            ordinal_features,
            nominal_features_encoded,
            interaction_features,
        ]
    )

    # Create coefficients DataFrame
    coefficients = pd.DataFrame(
        {"Feature": all_feature_names, "Coefficient": glm_model.coef_}
    )

    # Add feature type column
    coefficients["Feature_Type"] = "Interaction"
    coefficients.loc[
        coefficients["Feature"].isin(numerical_features), "Feature_Type"
    ] = "Numerical"
    coefficients.loc[coefficients["Feature"].isin(ordinal_features), "Feature_Type"] = (
        "Ordinal"
    )
    coefficients.loc[
        coefficients["Feature"].isin(nominal_features_encoded), "Feature_Type"
    ] = "Nominal"

    # Clean interaction feature names
    coefficients.loc[coefficients["Feature_Type"] == "Interaction", "Feature"] = (
        coefficients.loc[coefficients["Feature_Type"] == "Interaction", "Feature"]
        .str.replace(r"^x\d+\s", "", regex=True)
        .str.replace("_", " ")
    )

    # Sort by absolute coefficient value
    coefficients["Abs_Coefficient"] = abs(coefficients["Coefficient"])
    coefficients = coefficients.nlargest(top_n, "Abs_Coefficient")
    coefficients = coefficients[["Feature", "Feature_Type", "Coefficient"]]

    # Create rank index
    coefficients = coefficients.reset_index(drop=True)
    coefficients.index = coefficients.index + 1
    coefficients.index.name = "Rank"

    print(f"\nTop {top_n} Most Important Features")
    print("=" * 100)
    print("\nFeatures by Type:")
    print(coefficients["Feature_Type"].value_counts().to_string())
    print("\nDetailed Feature Importance Ranking:")
    print("=" * 100)

    return coefficients


def plot_learning_curve(
    estimator, X, y, scoring="neg_mean_squared_error", cv=5, n_jobs=-1
):
    """
    Plots a learning curve for a given estimator.

    Parameters:
    - estimator: The model to use for plotting the learning curve.
    - X: Feature matrix for training.
    - y: Target vector for training.
    - scoring: Scoring metric for evaluation.
    - cv: Number of cross-validation folds.
    - n_jobs: Number of jobs to run in parallel.
    """

    # Validate scoring parameter
    valid_scoring = {
        "r2",
        "recall",
        "accuracy",
        "precision",
        "neg_mean_squared_error",
        "f1",
    }
    if scoring not in valid_scoring:
        raise ValueError(f"Invalid scoring parameter. Must be one of {valid_scoring}")

    # Generate learning curve data
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=n_jobs,
    )

    # Calculate mean and standard deviation of training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training Score", marker="o")
    plt.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1
    )
    plt.plot(train_sizes, test_mean, label="Validation Score", marker="s")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

    # Add labels and legend
    plt.title("Learning Curve for LGBM Model")
    plt.xlabel("Training examples")
    plt.ylabel("RMSE Score")
    plt.legend()
    plt.grid()
    plt.show()
