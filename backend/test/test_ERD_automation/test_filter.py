import argparse
import sys
import os
import logging

# Add the backend directory to sys.path so we can import from app
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, backend_dir)
from app.ERD_automation.foreignKeyfilters import *
from app.ERD_automation.spider import find_inclusion_dependencies
from app.ERD_automation.primary_key_extraction import find_pks

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

    log_path = os.path.join(os.path.dirname(__file__), "TestingFilters.log")
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

    find_inclusion_dependencies()
    database.primary_keys=find_pks()
    logging.info(f"Length of Inclusion Dependencies: {len(database.filtered)}")
    primary_key_check()
    logging.info(f"Length of Inclusion Dependencies after primary key check: {len(database.filtered)}")
    null_check()
    logging.info(f"Length of Inclusion Dependencies after nan column check: {len(database.filtered)}")
    auto_incremental_pk_check()
    logging.info(f"Length of Inclusion Dependencies after sub sequence column check: {len(database.filtered)}")

if __name__ == "__main__":
    main()