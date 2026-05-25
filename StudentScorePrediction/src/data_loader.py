import sqlite3
from pathlib import Path

import pandas as pd


def validate_table_name(table_name: str) -> str:
    """
    Allow only simple SQLite identifiers from config.
    """
    if not table_name or not table_name.replace("_", "").isalnum():
        raise ValueError(f"Invalid table name: {table_name}")

    return table_name


def load_data(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Load dataset from a SQLite database table.

    Parameters
    ----------
    db_path : str
        Relative path to the SQLite database.
    table_name : str
        Name of the table to query.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at: {db_path}")

    safe_table_name = validate_table_name(table_name)

    with sqlite3.connect(db_path) as conn:
        query = f"SELECT * FROM {safe_table_name}"
        df = pd.read_sql_query(query, conn)

    return df
