import argparse
import os
import logging

import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.descriptionGeneration.descriptionGeneration import generate_description
from app.helper import database

from ..util import dummyDatabaseCreation

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--database_dir",
        type=str,
        default="/home/haseeb/Desktop/EKAI/ERD_automation/Dataset/train/northwind"
    )

    return parser.parse_args()

def main():
    """Main function to test and log the generation of database descriptions.

    This function performs the following steps:
      1. Sets up logging to a file named 'TestingDescriptionGeneration.log' in the current directory.
      2. Parses command-line arguments to obtain the path to the database directory.
      3. Loads CSV files from the specified database directory into the application's database.
      4. Generates a description of the database schema and content using an LLM.
      5. Logs the generated descriptions for each table in the database.

    Args:
        None

    Returns:
        None

    Side Effects:
        - Writes log output to 'TestingDescriptionGeneration.log'.
        - Modifies the global `database.db_frames` and `database.db_description` objects.

    Example:
        Run this script from the command line to generate and log database descriptions:
            $ python test_descriptionGeneration.py --database_dir /path/to/database
    """
    log_path = os.path.join(os.path.dirname(__file__), "TestingDescriptionGeneration.log")
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

    logging.info("Testing Dataset Description Generation")
    tables = database.db_description["tables"]
    for table_name, table_desc in tables.items():
        logging.info(f"{table_name} : {table_desc["note"]}")

if __name__ == "__main__":
    main()