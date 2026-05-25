# Student Score Prediction

## 1. Candidate Information

Name: `<Your full name as in application>`
Email: `<Your email address>`

## 2. Project Overview

This project predicts students' O-level mathematics examination scores using historical student performance data from U.A Secondary School. The goal is to identify weaker students before the examination so that additional academic support can be provided.

The project consists of two parts:

1. Exploratory Data Analysis in `eda.ipynb`
2. An end-to-end machine learning pipeline implemented in Python scripts under `src/`

## 3. Problem Statement

The target variable is `final_test`, the student's O-level mathematics examination score. Since `final_test` is a numerical score, this is formulated as a supervised regression problem.

The main business objective is not only to minimise prediction error, but also to support early identification of students who may require additional help.

## 4. Repository Structure

```text
.
├── configs/
│   └── config.yaml
├── data/
│   └── score.db
├── outputs/
│   ├── metrics.json
│   ├── best_model.joblib
│   └── error_analysis/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── error_analysis.py
│   ├── evaluate.py
│   ├── features.py
│   ├── models.py
│   ├── preprocessing.py
│   └── train.py
├── eda.ipynb
├── requirements.txt
└── README.md
```

The `score.db` file should be placed in `data/score.db` locally, but should not be submitted.

## 5. Setup Instructions

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the SQLite database at:

```text
data/score.db
```

## 6. How to Run the Pipeline

Train and evaluate the models:

```bash
python -m src.train
```

Run error analysis on the saved best model:

```bash
python -m src.error_analysis
```

The main configuration file is:

```text
configs/config.yaml
```

This file controls the database path, table name, target column, train-test split settings, and output paths.

## 7. Exploratory Data Analysis Summary

The EDA was used to understand the data before modelling and to guide preprocessing decisions.

Key findings:

1. The target variable `final_test` is numerical, so the task is a regression problem.
2. `final_test` and `attendance_rate` contain missing values.
3. Rows with missing `final_test` were removed because the target is unknown.
4. Missing numerical feature values are imputed inside the ML pipeline to avoid data leakage.
5. `index` and `student_id` are identifiers and were excluded from modelling.
6. Some categorical variables contain inconsistent labels, such as `tuition` having both `Yes`/`Y` and `No`/`N`.
7. `sleep_time` and `wake_time` were converted into numerical time-based features.
8. Several features showed predictive association with `final_test`, including `number_of_siblings`, `hours_per_week`, and `attendance_rate`.

The EDA findings were used to design the preprocessing and feature engineering steps in the ML pipeline.

## 8. Feature Processing

| Feature / Feature Type | Processing Applied | Reason |
|---|---|---|
| `final_test` | Used as target variable | Numerical exam score to predict |
| Missing `final_test` | Rows removed | Cannot train supervised model without target |
| `index`, `student_id` | Dropped | Identifier columns; risk of memorisation |
| Numerical features | Median imputation | Handles missing values robustly |
| Numerical features | StandardScaler | Useful for scale-sensitive models such as Ridge |
| Categorical features | Most frequent imputation + one-hot encoding | Converts categories into model-readable numerical format |
| `CCA`, `tuition` | Label cleaning | Fixes inconsistent category labels |
| `sleep_time`, `wake_time` | Converted into `sleep_hour`, `wake_hour`, `sleep_duration_hours` | Captures sleep duration and daily routine information |

All imputation, scaling, and one-hot encoding steps are fitted inside an sklearn `Pipeline` after the train-test split to reduce data leakage risk.

## 9. Modelling Approach

Four regression models were compared:

| Model | Purpose |
|---|---|
| DummyRegressor | Naive baseline that predicts the mean score |
| Ridge Regression | Simple regularised linear model |
| Random Forest Regressor | Nonlinear tree-based ensemble model |
| Gradient Boosting Regressor | Sequential ensemble model for nonlinear relationships |

The `DummyRegressor` provides a minimum benchmark. Ridge Regression tests whether a simple linear model is sufficient. Random Forest and Gradient Boosting were included to capture nonlinear relationships and feature interactions.

Hyperparameter tuning was performed for Random Forest and Gradient Boosting using `GridSearchCV` with 5-fold cross-validation on the training set only.

## 10. Development Process and Modelling Iteration

The pipeline was developed iteratively:

1. **Initial baseline modelling**  
   I first trained a `DummyRegressor`, Ridge Regression, Random Forest, and Gradient Boosting model using a shared preprocessing pipeline. This established a naive baseline and allowed comparison between linear and nonlinear models.

2. **Initial model comparison**  
   The Random Forest model achieved the best held-out test MAE, but its training error was much lower than its test error. This suggested that the model was fitting the training data more strongly than it generalised to unseen data.

3. **Hyperparameter tuning**  
   Because of the train-test performance gap, I added `GridSearchCV` for Random Forest and Gradient Boosting. The tuning focused on regularisation-related parameters such as tree depth, minimum samples per leaf, and number of estimators. Cross-validation was performed on the training set only.

4. **Error analysis**  
   After selecting the best model, I performed error analysis because the client objective is not only to minimise average prediction error, but also to identify weaker students. I therefore analysed students with actual `final_test < 50` and tested different predicted-score thresholds to understand the recall-precision trade-off.

5. **Feature importance analysis**  
   Finally, I inspected Random Forest feature importances to understand which features the model relied on. Feature importance was used for interpretation rather than automatic feature removal.

## 11. Evaluation Metrics

The models were evaluated using MAE, RMSE, and R².

| Metric | Interpretation |
|---|---|
| MAE | Average absolute prediction error in score points |
| RMSE | Penalises larger errors more heavily |
| R² | Proportion of variance explained compared with predicting the mean |

MAE was used as the primary model selection metric because it is directly interpretable as the average number of marks by which the prediction is wrong. RMSE was included to monitor large errors, and R² was included to measure explained variance.

## 12. Model Results

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| DummyRegressor | 11.61 | 13.93 | ~0.00 |
| Ridge Regression | 7.23 | 9.06 | 0.577 |
| Random Forest | 5.20 | 7.17 | 0.735 |
| Gradient Boosting | 5.91 | 7.70 | 0.695 |

Random Forest achieved the best test MAE of 5.20. This means that, on average, its predictions were approximately 5.2 marks away from the actual score.

Compared with the `DummyRegressor` baseline MAE of 11.61, the Random Forest model substantially improved predictive performance. However, the Random Forest training MAE was lower than its test MAE, suggesting some degree of overfitting remains.

## 13. Error Analysis

Since the client objective is to identify weaker students, additional error analysis was conducted for students with actual `final_test < 50`.

| Metric | Value |
|---|---:|
| Weak student threshold | 50 |
| Actual weak students in test set | 374 |
| Predicted weak students at threshold 50 | 273 |
| Missed weak students | 127 |
| Weak student recall | 0.660 |
| MAE for weak students | 5.30 |

At a predicted-score threshold of 50, the model identified around 66.0% of students who actually scored below 50. However, it missed 127 weak students. This suggests that although the model performs well overall, the operational threshold should be chosen carefully.

A threshold sensitivity analysis was also conducted:

| Prediction Threshold | Predicted At Risk | Recall | Precision |
|---:|---:|---:|---:|
| 45 | 106 | 0.273 | 0.962 |
| 50 | 273 | 0.660 | 0.905 |
| 55 | 527 | 0.858 | 0.609 |
| 60 | 851 | 0.936 | 0.411 |

Increasing the threshold improves recall but reduces precision. In a real deployment, the school may prefer a higher threshold such as 55 or 60 if the priority is to avoid missing weak students, even if this means providing support to some students who may not ultimately fail.

## 14. Feature Importance

The top Random Forest feature importances were:

| Feature | Importance |
|---|---:|
| number_of_siblings | 0.254 |
| hours_per_week | 0.169 |
| attendance_rate | 0.137 |
| n_male | 0.073 |
| n_female | 0.066 |
| CCA_None | 0.050 |
| direct_admission_Yes | 0.048 |

The model relied most heavily on `number_of_siblings`, `hours_per_week`, and `attendance_rate`. These are plausible predictors of student performance, but should be interpreted as associations rather than causal effects.

Feature importance was used mainly for interpretation rather than automatic feature removal. In future work, feature selection could be tested through ablation experiments and cross-validation.

## 15. Limitations and Future Work

Limitations:

1. The model is trained on historical observational data, so relationships should not be interpreted causally.
2. Some overfitting remains in the Random Forest model.
3. The weak-student threshold is a policy decision and should be agreed with the school.
4. Some features, such as `bag_color`, may be weak or noisy predictors.
5. The model has not been validated on a future cohort of students.

Future improvements:

1. Tune the intervention threshold with the school based on the cost of false negatives and false positives.
2. Perform ablation experiments to test whether removing low-value features improves generalisation.
3. Evaluate the model on a future cohort to test temporal generalisation.
4. Add model monitoring for prediction drift and performance drift if deployed.
5. Consider calibrated prediction intervals to communicate uncertainty in score predictions.
