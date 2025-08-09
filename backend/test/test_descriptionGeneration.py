import argparse
import os
import logging

import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.descriptionGeneration.descriptionGeneration import generate_description
from app.helper import database

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--database_dir",
        type=str,
        default="/home/haseeb/Desktop/EKAI/ERD_automation/Dataset/train/northwind"
    )

    return parser.parse_args()

def dummyDatabaseCreation(args):
    table_names = os.listdir(args.database_dir)
    for table_name in table_names:
        df = pd.read_csv(os.path.join(args.database_dir, table_name))
        database.db_frames[table_name.split(".")[0]] = df

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

    dummyDatabaseCreation(args=args)
    generate_description()

    logging.info("Testing Dataset Description Generation")
    logging.info(database.db_description)  

if __name__ == "__main__":
    main()