import pandas as pd
import json
from redis import Redis
import os

class Database:
    """A class for managing database tables, ground truth data, and descriptions.

    This class provides a structure to store multiple pandas DataFrames representing
    database tables, ground truth (GT) data loaded from JSON, and generated descriptions
    for the database tables.

    Attributes:
        gt (dict): 
            A dictionary containing ground truth typically loaded from a JSON file.
        db_description (dict): 
            A dictionary containing generated descriptions for the database tables.
    """

    def __init__(self):
        """Initializes the Database object with empty data structures."""
        self.gt: dict = {}
        self.db_description: dict = {}
        
        # Use environment variables for Redis connection
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        self.r = Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.inclusion_dependencies = [] # (reference_attribute, dependent_attribute) table_name.column_name
        self.filtered = []
        self.primary_keys = {} #table name: (tableName.AttributeName, score)

    def store_dataframe_in_redis(self, table_name: str, df: pd.DataFrame):
        """Store a DataFrame in Redis using an improved method that preserves all rows.
        
        This method stores each row as a separate JSON string in a Redis list.
        
        Args:
            table_name (str): Name of the table
            df (pd.DataFrame): DataFrame to store
        """
        
        # Store each row as a JSON string in a Redis list
        for _, row in df.iterrows():
            row_json = json.dumps(row.to_dict())
            self.r.lpush(table_name, row_json)
