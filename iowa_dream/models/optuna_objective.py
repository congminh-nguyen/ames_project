import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold


def optuna_objective(trial, X, y, pipeline):
    # Define the hyperparameter search space
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 30, 300, step=30),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 5.0),
    }

    # Update the pipeline with dynamic LGBMRegressor parameters
    pipeline.set_params(
        estimator__n_estimators=param_grid["n_estimators"],
        estimator__learning_rate=param_grid["learning_rate"],
        estimator__num_leaves=param_grid["num_leaves"],
        estimator__min_child_weight=param_grid["min_child_weight"],
    )

    # Cross-validation for the pipeline
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Predict and calculate RMSE
        preds = pipeline.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        cv_scores.append(rmse)

    # Return the mean RMSE across folds
    return np.mean(cv_scores)
