import argparse
import os
import logging

import pandas as pd

import sys
import os

# Add the backend directory to sys.path so we can import from app
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, backend_dir)

from app.helper import hdbscan, database
from app.descriptionGeneration.descriptionGeneration import generate_description

# Add the test directory to sys.path for util imports
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, test_dir)

from util import log_clusters, log_description, dummyDatabaseCreation


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--database_dir",
        type=str,
        default="/home/haseeb/Desktop/EKAI/ERD_automation/Dataset/train/northwind"
    )

    return parser.parse_args()

def main():
    """Main function to test and log the compartmentalization (clustering) of database table descriptions.

    This function performs the following steps:
      1. Sets up logging to a file named 'TestingCompartmentalization.log' in the current directory.
      2. Parses command-line arguments to obtain the path to the database directory.
      3. Loads CSV files from the specified database directory into the application's database.
      4. Generates a description of the database schema and content using an LLM.
      5. Prepares a list of table descriptions for clustering.
      6. Applies HDBSCAN clustering to the table descriptions.
      7. Logs the generated descriptions and the resulting clusters.

    Args:
        None

    Returns:
        None

    Side Effects:
        - Writes log output to 'TestingCompartmentalization.log'.
        - Modifies the global `database.db_frames` and `database.db_description` objects.
        - Logs the clusters and descriptions using the provided utility functions.

    Example:
        Run this script from the command line to test compartmentalization:
            $ python test_compartmentalization.py --database_dir /path/to/database
    """
    log_path = os.path.join(os.path.dirname(__file__), "TestingCompartmentalization.log")
    logging.basicConfig(
                    filename=log_path,
                    encoding="utf-8",
                    filemode="w",
                    format="{asctime} - {levelname} - {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M",
                    level=logging.INFO
    )  

    args = parse_args()

    dummyDatabaseCreation(database_dir=args.database_dir)
    generate_description()

    table_names = list(database.db_description["tables"].keys())
    texts = [f"{table_name} : {database.db_description['tables'][table_name]['note']}" for table_name in table_names]  

    clusters = hdbscan.cluster(texts)
    
    log_description()
    log_clusters(clusters=clusters)

if __name__ == "__main__":
    main()