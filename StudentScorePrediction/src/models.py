from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


def get_models(random_state: int = 42):
    """
    Return a dictionary of regression models to compare.

    Includes:
    - DummyRegressor as a baseline
    - Ridge Regression as a simple linear model
    - Random Forest as a nonlinear tree-based model
    - Gradient Boosting as a stronger boosting model
    """
    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=random_state,
        ),
    }

    return models


'''
I included DummyRegressor as a naive baseline to check whether the models learn more than simply predicting the average score. I then used Ridge Regression as a simple linear ML baseline, followed by tree-based ensemble models to capture nonlinear relationships.
'''