from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, median_absolute_error, r2_score
import pandas as pd
import numpy as np

def reevaluate_models(models, X, y):
    """Evaluate multiple models on a test set and return formatted results.
    
    Parameters
    ----------
    models : list
        List of fitted model objects with predict method
    X : array-like
        Test features
    y : array-like
        True target values
        
    Returns
    -------
    pd.io.formats.style.Styler
        Styled DataFrame containing evaluation metrics for each model
    """
    results = {
        'Model': [],
        'RMSE': [], 
        'RMSED': [], # Root Mean Squared Error Degradation
        'MAPE': [],
        'MedAE': [],
        'R-squared': []
    }
    
    for model in models:
        y_pred = model.predict(X)
        
        results['Model'].append(model.__class__.__name__)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        results['RMSE'].append(rmse)
        results['RMSED'].append(rmse / np.mean(y)) # Normalized RMSE
        results['MAPE'].append(mean_absolute_percentage_error(y, y_pred))
        results['MedAE'].append(median_absolute_error(y, y_pred))
        results['R-squared'].append(r2_score(y, y_pred))
    
    results_df = pd.DataFrame(results)
    results_df.set_index('Model', inplace=True)
    
    # Only highlight if there's more than one model
    if len(models) > 1:
        styled_df = results_df.style.format({
            'RMSE': '{:.2f}',
            'RMSED': '{:.2%}',
            'MAPE': '{:.2%}',
            'MedAE': '{:.2f}',
            'R-squared': '{:.2f}'
        }).highlight_min(subset=['RMSE'], color='lightgreen').highlight_max(subset=['RMSE'], color='lightcoral')
    else:
        styled_df = results_df.style.format({
            'RMSE': '{:.2f}',
            'RMSED': '{:.2%}',
            'MAPE': '{:.2%}',
            'MedAE': '{:.2f}',
            'R-squared': '{:.2f}'
        })
        
    return styled_df.set_caption("Model Evaluation Metrics")