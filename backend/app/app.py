from fastapi import FastAPI, UploadFile, Depends
from fastapi.encoders import jsonable_encoder
import pandas as pd
from io import StringIO
import json
from .Database import Database
from typing import Optional, Dict, List
from fastapi import UploadFile
from .helper import database, hdbscan 
from .descriptionGeneration.descriptionGeneration import generate_description
import os
import logging
from .ERD_automation.spider import find_inclusion_dependencies
import csv

app = FastAPI()

log_path = os.path.join(os.path.dirname(__file__), "log.log")
logging.basicConfig(
                    filename=log_path,
                    encoding="utf-8",
                    filemode="w",
                    format="{asctime} - {levelname} - {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M",
                    level=logging.INFO
)

@app.post("/upload-database-file")
async def upload_db_file(
    file: UploadFile
) -> dict:
    """Uploads a CSV file and appends its contents as a DataFrame to the database.

    Args:
        file (UploadFile): The uploaded CSV file.
        database (Database): The database instance.

    Returns:
        dict: A message indicating the backend is working.
    """
    
    contents: bytes = await file.read()
    decoded: str = contents.decode("utf-8")
    df: pd.DataFrame = pd.read_csv(StringIO(decoded))
    table_name = file.filename.rsplit(".",1)[0]
    database.store_dataframe_in_redis(table_name=table_name, df = df)

    return {
            "message": "Database file uploaded successfully",
            "filename": file.filename,
            "table_name": table_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)
    }

@app.post("/upload-ground-truth")
async def upload_gt_file(
    file: UploadFile
) -> dict:
    """Uploads a ground truth JSON file and stores its contents in the database.

    Args:
        file (UploadFile): The uploaded JSON file.
        database (Database): The database instance.

    Returns:
        dict: Status and keys of the uploaded ground truth data.
    """
    generate_description() # All database files are uplaoded so generate databse descriptions 
    contents: bytes = await file.read()
    decoded: str = contents.decode("utf-8")
    gt_data: dict = json.loads(decoded)
    database.gt = gt_data
    
    return {
            "status": "success",
            "filename": file.filename,
            "keys": list(gt_data.keys()),
            "description_generated": "error" not in database.db_description
    }

@app.get("/database-description")
def get_database_description():
    return database.db_description

@app.get("/reset-database")
def reset_database():
    database.db_description.clear()
    database.gt.clear()
    return {
            "status": "success",
            "message": "Database reset successfully",
            "cleared": {
                "ground_truth": True,
                "descriptions": True
            }
    }

@app.get("/cluster")
async def cluster(cluster_method: str) -> dict:
    """Performs clustering on the database tables using the specified clustering method.

    This endpoint generates descriptions for all tables in the database, constructs
    a list of text representations for each table, and applies the selected clustering
    algorithm to group the tables based on their semantic similarity.

    Args:
        cluster_method (str): The clustering method to use. Supported values include:
            - "hdbScan": Uses the HDBSCAN clustering algorithm.

    Returns:
        dict: The result of the clustering operation, as returned by the selected
            clustering algorithm. The structure of the result depends on the
            clustering method used.

    Raises:
        KeyError: If the specified clustering method is not supported.

    Example:
        GET /cluster?cluster_method=hdbScan
    """
    table_names = list(database.db_description["tables"].keys())
    texts = [
        f"{table_name} : {database.db_description['tables'][table_name]['note']}"
        for table_name in table_names
    ]

    clusteror_map: dict[str, callable] = {
        "hdbScan": hdbscan.cluster
    }

    if cluster_method not in clusteror_map:
            raise ValueError(f"Method '{cluster_method}' not supported.")
           
    
    result = clusteror_map[cluster_method](text=texts)
    # Ensure numpy types are converted to native Python types
    return jsonable_encoder(result)

@app.get("/saveInclDependencies")
async def saveInclDependencies():
    find_inclusion_dependencies()
