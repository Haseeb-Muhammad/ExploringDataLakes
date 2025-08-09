import sys
import os
from pathlib import Path
from typing import List, Union, Optional, Dict
import logging
import numpy as np
from collections import defaultdict, Counter
from ...util import log_clusters

# Add the backend directory to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[3]))

# Now import after setting the path
from app.compartmentalization.clustering.clusterors.HDBScan import HDBScan
from app.compartmentalization.clustering.clusterors.Raptor import RAPTORClusteror
from app.compartmentalization.clustering.encoders.sentence_transformer import SentenceTransformerEncoder

import argparse
import json
import logging
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--description_path",
    type=str,
    default="/home/haseeb/Desktop/EKAI/ExploringDataLakes/backend/test/compartmentalization/clustering/northwind_descriptions.json"
    )

    return parser.parse_args()

def main():
    """Main function to perform clustering on table descriptions and log the results.

    This function executes the following steps:
      1. Sets up logging to a file named 'logs.log' in the current directory.
      2. Parses command-line arguments to obtain the path to a JSON file containing table descriptions.
      3. Loads table descriptions from the specified JSON file.
      4. Prepares a list of table descriptions for clustering.
      5. Initializes a sentence transformer encoder and an HDBScan clusteror.
      6. Logs information about the clusteror and encoder being used.
      7. Applies the clustering algorithm to the table descriptions.
      8. Logs the resulting clusters using a utility function.

    Args:
        None

    Returns:
        None

    Side Effects:
        - Writes log output to 'logs.log'.
        - Reads from the file specified by the '--description_path' argument.
        - Logs the clusters and information about the clustering process.

    Example:
        Run this script from the command line to cluster table descriptions:
            $ python test_clusteror.py --description_path /path/to/descriptions.json
    """
    log_path = os.path.join(os.path.dirname(__file__), "logs.log")
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

    with open(args.description_path) as f: 
        data = json.load(f)
        table_names = list(data["tables"].keys())
        texts = [f"{table_name} : {data['tables'][table_name]['note']}" for table_name in table_names]  

    encoder = SentenceTransformerEncoder(model_name="all-MiniLM-L6-v2")
    clusteror = HDBScan(encoder=encoder)
    
    logging.info(f"{'-'*100}")
    logging.info(f"Clusteror: {str(clusteror)}")
    logging.info(f"Encoder: {str(encoder)}")
    logging.info(f"{'-'*100}")

    clusters = clusteror.cluster(text=texts)
    log_clusters(clusters=clusters)

if __name__ == "__main__":
    main()