import argparse
import sys
import os
import logging

# Add the backend directory to sys.path so we can import from app
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, backend_dir)

from app.ERD_automation.spider import find_inclusion_dependencies
from app.helper import database

# Add the test directory to sys.path for util imports
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, test_dir)

from util import dummyDatabaseCreation

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
    find_inclusion_dependencies(output_file="logging.txt")

if __name__ == "__main__":
    main()