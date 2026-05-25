import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.features import prepare_features
from src.evaluate import evaluate_regression


def get_feature_names(pipeline):
    """
    Extract feature names after preprocessing.

    Works for a pipeline with:
    - preprocessor: ColumnTransformer
    - model: trained estimator
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def get_feature_importance(pipeline):
    """
    Extract feature importance from models that support feature_importances_.
    Example: RandomForestRegressor, GradientBoostingRegressor.
    """
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return None

    feature_names = get_feature_names(pipeline)
    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    return importance_df


def run_error_analysis(config_path: str = "configs/config.yaml"):
    """
    Run error analysis on the saved best model.

    Outputs:
    - test_predictions.csv
    - largest_errors.csv
    - weak_student_analysis.json
    - feature_importance.csv, if model supports feature importance
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = Path(config["output"]["model_path"])

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. Run `python -m src.train` first."
        )

    # Load data
    df = load_data(
        db_path=config["data"]["db_path"],
        table_name=config["data"]["table_name"],
    )

    # Prepare features and target using the same logic as training
    X, y = prepare_features(
        df=df,
        target_col=config["data"]["target_col"],
        drop_cols=config["data"]["drop_cols"],
    )

    # Recreate same train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    # Load best model
    pipeline = joblib.load(model_path)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Build prediction results dataframe
    results_df = X_test.copy()
    results_df["actual_final_test"] = y_test
    results_df["predicted_final_test"] = y_pred
    results_df["residual"] = results_df["actual_final_test"] - results_df["predicted_final_test"]
    results_df["absolute_error"] = results_df["residual"].abs()

    # Overall metrics
    metrics = evaluate_regression(y_test, y_pred)
    metrics = {key: float(value) for key, value in metrics.items()}

    # Largest errors
    largest_errors = results_df.sort_values("absolute_error", ascending=False).head(20)

    # Weak student analysis
    # Threshold can be adjusted, but 50 is a sensible pass/fail-style cutoff.
    weak_threshold = 50

    weak_mask = results_df["actual_final_test"] < weak_threshold
    predicted_weak_mask = results_df["predicted_final_test"] < weak_threshold

    weak_students = results_df[weak_mask]
    missed_weak_students = results_df[weak_mask & ~predicted_weak_mask]

    weak_student_analysis = {
        "weak_threshold": weak_threshold,
        "n_actual_weak_students": int(weak_mask.sum()),
        "n_predicted_weak_students": int(predicted_weak_mask.sum()),
        "n_missed_weak_students": int(len(missed_weak_students)),
        "weak_student_recall": (
            float((weak_mask & predicted_weak_mask).sum() / weak_mask.sum())
            if weak_mask.sum() > 0
            else None
        ),
        "mean_absolute_error_for_weak_students": (
            float(weak_students["absolute_error"].mean())
            if len(weak_students) > 0
            else None
        ),
    }

    # Save outputs
    output_dir = Path("outputs/error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "test_predictions.csv", index=False)
    largest_errors.to_csv(output_dir / "largest_errors.csv", index=False)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(output_dir / "weak_student_analysis.json", "w") as f:
        json.dump(weak_student_analysis, f, indent=4)

    # Feature importance
    importance_df = get_feature_importance(pipeline)

    if importance_df is not None:
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        print("\nTop 15 feature importances:")
        print(importance_df.head(15))
    else:
        print("\nFeature importance is not available for this model type.")

    print("\nOverall test metrics:")
    print(metrics)

    print("\nWeak student analysis:")
    print(weak_student_analysis)

    print(f"\nSaved error analysis outputs to: {output_dir}")

    threshold_results = []

    for threshold in [45, 50, 55, 60]:
        actual_weak = results_df["actual_final_test"] < 50
        predicted_at_risk = results_df["predicted_final_test"] < threshold

        true_positives = (actual_weak & predicted_at_risk).sum()
        false_negatives = (actual_weak & ~predicted_at_risk).sum()
        false_positives = (~actual_weak & predicted_at_risk).sum()

        recall = true_positives / actual_weak.sum() if actual_weak.sum() > 0 else None
        precision = (
            true_positives / predicted_at_risk.sum()
            if predicted_at_risk.sum() > 0
            else None
        )

        threshold_results.append(
            {
                "prediction_threshold": threshold,
                "n_predicted_at_risk": int(predicted_at_risk.sum()),
                "true_positives": int(true_positives),
                "false_negatives": int(false_negatives),
                "false_positives": int(false_positives),
                "recall": float(recall) if recall is not None else None,
                "precision": float(precision) if precision is not None else None,
            }
        )

    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv(output_dir / "threshold_sensitivity.csv", index=False)

    print("\nThreshold sensitivity analysis:")
    print(threshold_df)

if __name__ == "__main__":
    run_error_analysis()