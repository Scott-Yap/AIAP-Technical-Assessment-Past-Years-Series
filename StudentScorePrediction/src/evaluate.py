import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression predictions using MAE, RMSE, and R².

    MAE:
        Average absolute error in score points.

    RMSE:
        Penalises larger errors more heavily.

    R²:
        Proportion of variance explained by the model.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
