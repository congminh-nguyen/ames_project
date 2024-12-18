from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def reevaluate_models(models, X, y, model_names=None) -> pd.DataFrame:
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

    results_df = pd.DataFrame(results)
    results_df.set_index("Model", inplace=True)

    # Format numeric columns
    for col in results_df.columns:
        if col == "RMSE" or col == "MedAE" or col == "R-squared":
            results_df[col] = results_df[col].map("{:.2f}".format)
        else:
            results_df[col] = results_df[col].map("{:.2%}".format)

    return results_df
