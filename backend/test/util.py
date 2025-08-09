import logging
from typing import Dict, List
from app.helper import database
import os
import pandas as pd

def log_clusters(clusters: dict):
    """Logs the clusters and their contents using the logging module.

    Args:
        clusters (dict): A dictionary representing the clusters. The expected format is:
            {
                cluster_level: {
                    clusterID: [list of descriptions (str)]
                }
            }

    Side Effects:
        Writes cluster information to the log using the logging module.

    Example:
        log_clusters({1: {0: ['table1: desc', 'table2: desc'], 1: ['table3: desc']}})
    """
    logging.info(f"{'-'*50}")
    logging.info(f"Logging Clusters")
    for cluster_level, val in clusters.items():
        logging.info(f"Cluster Level {cluster_level}")
        for clusterID, cluster in val.items():
            logging.info(f"Cluster {clusterID}: {[desc.split(':')[0] for desc in cluster]}")
        logging.info(f"{'-'*20}")
    logging.info(f"{'-'*50}")

def log_description():
    """Logs the generated descriptions for each table in the database.

    Side Effects:
        Writes table names and their notes to the log using the logging module.

    Example:
        log_description()
    """
    logging.info(f"{'-'*50}")
    logging.info("Testing Dataset Description Generation")
    tables = database.db_description["tables"]
    for table_name, table_desc in tables.items():
        logging.info(f"{table_name} : {table_desc['note']}")
    logging.info(f"{'-'*50}")

def dummyDatabaseCreation(database_dir):
    """Loads CSV files from the specified directory into the database's frame dictionary.

    Args:
        database_dir (str): The path to the directory containing CSV files representing tables.

    Side Effects:
        Populates the `database.db_frames` dictionary with DataFrames, using the file name
        (without extension) as the key.

    Example:
        dummyDatabaseCreation('/path/to/database_dir')
    """
    table_names = os.listdir(database_dir)
    for table_name in table_names:
        df = pd.read_csv(os.path.join(database_dir, table_name))
        database.db_frames[table_name.split(".")[0]] = df
