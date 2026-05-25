from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_types(X):
    """
    Identify numerical and categorical columns from the feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    tuple[list[str], list[str]]
        Numerical columns and categorical columns.
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    return numeric_cols, categorical_cols


def build_preprocessor(X):
    """
    Build a ColumnTransformer for preprocessing numerical and categorical features.

    Numerical features:
    - Impute missing values using median
    - Standardise using StandardScaler

    Categorical features:
    - Impute missing values using most frequent category
    - One-hot encode categories

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    ColumnTransformer
        Sklearn preprocessing transformer.
    """
    numeric_cols, categorical_cols = get_feature_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


'''I used a ColumnTransformer to apply different preprocessing steps to numerical and categorical features. Numerical features were median-imputed and standardised. Categorical features were imputed with the most frequent category and one-hot encoded. This preprocessing is placed inside the sklearn pipeline so that transformations are fitted only on the training data, reducing data leakage risk.'''

'''Numerical features were standardised primarily for scale-sensitive models such as Ridge Regression. Tree-based models such as Random Forest are generally not sensitive to feature scaling, but using the same preprocessing pipeline across models keeps the comparison simple and consistent.
'''
