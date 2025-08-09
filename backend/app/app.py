from fastapi import FastAPI, UploadFile, Depends
import pandas as pd
from io import StringIO
import json
from .Database import Database
from typing import Optional, Dict, List
from fastapi import UploadFile
from helper import database, sentence_transformer, hdbscan
from descriptionGeneration.descriptionGeneration import generate_description

app = FastAPI()

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
    database.db_frames[file.filename] = df
    return {"message": "Backend is working"}

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
    contents: bytes = await file.read()
    decoded: str = contents.decode("utf-8")
    gt_data: dict = json.loads(decoded)
    database.gt = gt_data
    return {"status": "success", "keys": list(gt_data.keys())}

@app.get("/HDBScanClustering")
async def HDBScanClustering()-> Dict[int, Dict[int, List]]:
    generate_description()
    return hdbscan.cluster(database.db_description)

