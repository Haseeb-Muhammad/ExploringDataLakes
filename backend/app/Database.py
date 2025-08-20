import pandas as pd
import json
from redis import Redis
import os

class Database:

    def __init__(self):
        """
        Initializes the Database object by setting up internal data structures and establishing a Redis connection.
        Attributes:
            gt (dict): Stores ground truth or related data.
            db_description (dict): Contains descriptions or metadata about the database.
            r (Redis): Redis client instance for database operations, configured using environment variables 'REDIS_HOST' and 'REDIS_PORT'.
            inclusion_dependencies (list): List of tuples representing inclusion dependencies in the format (reference_attribute, dependent_attribute).
            filtered (list): List of tuples representing filtered inclusion dependencies.
            primary_keys (dict): Maps table names to their primary key attributes and associated scores.
        """
        self.gt: dict = {}
        self.db_description: dict = {}
                                
        # Use environment variables for Redis connection
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        self.r = Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.inclusion_dependencies = [] # (reference_attribute, dependent_attribute) table_name.column_name
        self.filtered = [] # (reference_attribute, dependent_attribute) table_name.column_name
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
