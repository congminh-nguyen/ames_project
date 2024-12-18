import numpy as np


# Define the custom regression loss function
def custom_regression_loss(y_true, y_pred):
    # Calculate the residual (error)
    residual = y_pred - y_true
    # Calculate the gradient (first-order derivative)
    gradient = 2 * residual
    # Calculate the hessian (second-order derivative)
    hessian = 2 * np.ones_like(y_true)
    # Define a simple linear penalty term
    penalty = np.abs(residual) * 0.05
    # Combine the gradient and penalty
    gradient += penalty
    return gradient, hessian
