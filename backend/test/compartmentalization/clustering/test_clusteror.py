import sys
import os
from pathlib import Path
from typing import List, Union, Optional, Dict
import logging
import numpy as np
from collections import defaultdict, Counter

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

def log_clusters(clusters):
    for cluster_level, val in clusters.items():
        logging.info(f"Cluster Level {cluster_level}")
        for clusterID, cluster in val.items():
            logging.info(f"Cluster {clusterID}: {[desc.split(":")[0] for desc in cluster]}")
        logging.info(f"{'-'*50}")

def main():

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
        texts = [f"{table_name} : {data["tables"][table_name]['note']}" for table_name in table_names]  

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