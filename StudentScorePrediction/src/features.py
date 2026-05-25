import pandas as pd


def clean_categorical_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean inconsistent categorical labels found during EDA.
    """
    df = df.copy()

    if "CCA" in df.columns:
        df["CCA"] = df["CCA"].replace(
            {
                "CLUBS": "Clubs",
                "SPORTS": "Sports",
                "ARTS": "Arts",
                "NONE": "None",
            }
        )

    if "tuition" in df.columns:
        df["tuition"] = df["tuition"].replace(
            {
                "Y": "Yes",
                "N": "No",
            }
        )

    return df


def time_to_minutes(time_str: str) -> int:
    """
    Convert HH:MM time string to minutes after midnight.
    """
    hour, minute = map(int, time_str.split(":"))
    return hour * 60 + minute


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer numerical time features from sleep_time and wake_time.
    """
    df = df.copy()

    df["sleep_minutes"] = df["sleep_time"].apply(time_to_minutes)
    df["wake_minutes"] = df["wake_time"].apply(time_to_minutes)

    df["sleep_hour"] = df["sleep_minutes"] / 60
    df["wake_hour"] = df["wake_minutes"] / 60

    df["sleep_duration_hours"] = (
        (df["wake_minutes"] - df["sleep_minutes"]) % (24 * 60)
    ) / 60

    df = df.drop(
        columns=["sleep_time", "wake_time", "sleep_minutes", "wake_minutes"]
    )

    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target vector y.
    """
    df = df.copy()

    df = df.dropna(subset=[target_col])
    df = clean_categorical_labels(df)
    df = add_time_features(df)

    y = df[target_col]
    X = df.drop(columns=[target_col] + drop_cols)

    return X, y
