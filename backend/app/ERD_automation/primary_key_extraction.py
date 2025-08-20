from ..helper import database
import json
import pandas as pd
from .attribute import Attribute

def find_pks():
    """Identifies primary key candidates for each table in the database.

    This function retrieves all table names from the Redis database, loads their rows,
    and analyzes each column to determine its suitability as a primary key using the
    Attribute class. The column with the highest primary key score (`pkScore`) for each
    table is selected as the primary key candidate.

    Returns:
        dict: A dictionary mapping each table name to a tuple containing the full
            column name (in the format "table.column") and its primary key score.

    """
    table_names = database.r.keys('*')
    attributes = {}

    for table_name in table_names:
        list_items = database.r.lrange(table_name, 0, -1)
        if not list_items:
            continue

        # Use list comprehension for faster row parsing
        rows_data = [json.loads(item) for item in list_items]
        if not rows_data:
            continue

        df = pd.DataFrame(rows_data)

        # Use enumerate and vectorized operations
        for i, column in enumerate(df.columns):
            # Dropna and convert to string in one go, avoid .tolist() if possible
            non_null_values = df[column].dropna().astype(str).values
            if non_null_values.size > 0:
                attr = Attribute(table_name, column, non_null_values)
                attr.position = 1 / (i + 1)
                attributes[f"{table_name}.{column}"] = attr

    pk_table = {}
    for key, value in attributes.items():
        table_name = key.split(".")[0]
        current_pk = pk_table.get(table_name)
        if not current_pk or current_pk[1] < value.pkScore:
            pk_table[table_name] = (value.fullName, value.pkScore)

    return pk_table