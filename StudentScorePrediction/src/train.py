import json
from pathlib import Path

import joblib
import yaml
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import load_data
from src.evaluate import evaluate_regression
from src.features import prepare_features
from src.models import get_models
from src.preprocessing import build_preprocessor


def to_serializable_metrics(metrics: dict) -> dict:
    """
    Convert metric values to Python floats so they can be saved as JSON.
    """
    return {key: float(value) for key, value in metrics.items()}


def to_serializable_params(params):
    """
    Convert GridSearchCV params into JSON-serialisable values.
    """
    if params is None:
        return None

    serializable = {}

    for key, value in params.items():
        if hasattr(value, "item"):
            serializable[key] = value.item()
        else:
            serializable[key] = value

    return serializable


def get_param_grids():
    """
    Hyperparameter grids for selected models.

    Parameter names use the 'model__' prefix because the estimator is inside
    an sklearn Pipeline step named 'model'.
    """
    return {
        "random_forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [8, 12, None],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 1.0],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__min_samples_leaf": [1, 3],
        },
    }


def train(config_path: str = "configs/config.yaml"):
    """
    Train and evaluate multiple regression models.

    The pipeline:
    1. Loads data from SQLite
    2. Prepares features and target
    3. Splits data into train/test sets
    4. Fits preprocessing and model inside an sklearn Pipeline
    5. Tunes selected models using cross-validation on the training set
    6. Evaluates each model on train and test data
    7. Selects the best model by lowest test MAE
    8. Saves metrics and the best fitted pipeline
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    df = load_data(
        db_path=config["data"]["db_path"],
        table_name=config["data"]["table_name"],
    )

    X, y = prepare_features(
        df=df,
        target_col=config["data"]["target_col"],
        drop_cols=config["data"]["drop_cols"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    models = get_models(random_state=config["models"]["random_state"])
    param_grids = get_param_grids()

    results = {}
    fitted_pipelines = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        preprocessor = build_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        best_params = None
        best_cv_mae = None
        tuned = False

        if model_name in param_grids:
            print(f"Tuning {model_name} with GridSearchCV...")

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[model_name],
                scoring="neg_mean_absolute_error",
                cv=5,
                n_jobs=-1,
                verbose=1,
            )

            search.fit(X_train, y_train)

            best_pipeline = search.best_estimator_
            best_params = to_serializable_params(search.best_params_)
            best_cv_mae = float(-search.best_score_)
            tuned = True

            print(f"Best params for {model_name}: {best_params}")
            print(f"Best CV MAE for {model_name}: {best_cv_mae:.4f}")

        else:
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline

        y_train_pred = best_pipeline.predict(X_train)
        y_test_pred = best_pipeline.predict(X_test)

        train_metrics = to_serializable_metrics(
            evaluate_regression(y_train, y_train_pred)
        )
        test_metrics = to_serializable_metrics(
            evaluate_regression(y_test, y_test_pred)
        )

        results[model_name] = {
            "tuned": tuned,
            "best_params": best_params,
            "best_cv_mae": best_cv_mae,
            "train": train_metrics,
            "test": test_metrics,
        }

        fitted_pipelines[model_name] = best_pipeline

        print(f"{model_name} train: {train_metrics}")
        print(f"{model_name} test:  {test_metrics}")

    best_model_name = min(
        results,
        key=lambda name: results[name]["test"]["mae"],
    )

    best_pipeline = fitted_pipelines[best_model_name]
    best_test_mae = results[best_model_name]["test"]["mae"]

    print(f"\nBest model: {best_model_name}")
    print(f"Best test MAE: {best_test_mae:.4f}")

    metrics_path = Path(config["output"]["metrics_path"])
    model_path = Path(config["output"]["model_path"])

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "best_model": best_model_name,
        "selection_metric": "test_mae",
        "note": "Hyperparameter tuning, where applied, used cross-validation on the training set only.",
        "results": results,
    }

    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=4)

    joblib.dump(best_pipeline, model_path)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved best model to: {model_path}")


if __name__ == "__main__":
    train()