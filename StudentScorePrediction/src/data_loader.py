import sqlite3
from pathlib import Path

import pandas as pd


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

    with sqlite3.connect(db_path) as conn:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)

    return df
