import argparse
import os
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.helper import database

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
    print(database.extract_schema_from_redis())
    
if __name__ == "__main__":
    main()