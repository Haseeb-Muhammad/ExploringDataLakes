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
from app.compartmentalization.clustering.clusteror.HDBScan import HDBScan
from app.compartmentalization.clustering.clusteror.Raptor import RAPTORClusteror
from app.compartmentalization.clustering.encoder.sentence_transformer import SentenceTransformerEncoder

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

def log_clusters(table_names : list[str], labels : list[int]):
    table_clusters = {table_names[i]: int(labels[i]) for i in range(len(table_names))}
    cluster_groups = defaultdict(list)
    for table, label in table_clusters.items():
        cluster_groups[label].append(table)

    for label, tables in cluster_groups.items():
        logging.info(f"\nCluster {label}:")
        for t in tables:
            logging.info(f" - {t}")

def log_hierarchical_clusters(
    cluster_output: Union[List[int], Dict[str, Union[List[int], np.ndarray]]],
    table_names: List[str],
    logger_name: str = "ClusterAnalyzer"
) -> None:
    """
    Log hierarchical cluster information with table names mapping.
    
    Args:
        cluster_output: Output from the cluster function - either simple labels list
                       or dict with hierarchical information
        table_names: List of table names corresponding to the embeddings order
        logger_name: Name for the logger instance
    """
    logger = logging.getLogger(logger_name)
    
    # Validate inputs
    if not table_names:
        logger.error("Empty table_names list provided")
        return
    
    # Handle both simple labels and hierarchical output
    if isinstance(cluster_output, list):
        labels = cluster_output
        linkage_matrix = None
        multi_labels = None
        is_hierarchical = False
    elif isinstance(cluster_output, dict):
        labels = cluster_output.get('labels', [])
        linkage_matrix = cluster_output.get('linkage_matrix', None)
        multi_labels = cluster_output.get('multi_labels', None)
        is_hierarchical = True
    else:
        logger.error(f"Invalid cluster_output type: {type(cluster_output)}")
        return
    
    # Validate lengths match
    if len(labels) != len(table_names):
        logger.error(f"Length mismatch: {len(labels)} labels vs {len(table_names)} table names")
        return
    
    logger.info("=" * 80)
    logger.info("HIERARCHICAL CLUSTER ANALYSIS REPORT")
    logger.info("=" * 80)
    
    # Basic statistics
    unique_labels = set(labels)
    num_clusters = len(unique_labels)
    num_tables = len(table_names)
    
    logger.info(f"Total Tables Analyzed: {num_tables}")
    logger.info(f"Number of Clusters Found: {num_clusters}")
    logger.info(f"Clustering Type: {'Hierarchical' if is_hierarchical else 'Simple'}")
    
    if -1 in unique_labels:
        noise_count = labels.count(-1)
        logger.info(f"Unclustered/Noise Tables: {noise_count}")
        unique_labels.remove(-1)
    
    logger.info("-" * 80)
    
    # Detailed cluster analysis
    cluster_stats = Counter(labels)
    logger.info("CLUSTER SIZE DISTRIBUTION:")
    for cluster_id in sorted(unique_labels):
        size = cluster_stats[cluster_id]
        percentage = (size / num_tables) * 100
        logger.info(f"  Cluster {cluster_id}: {size} tables ({percentage:.1f}%)")
    
    logger.info("-" * 80)
    logger.info("DETAILED CLUSTER MEMBERSHIP:")
    
    # Group tables by cluster
    clusters_dict = defaultdict(list)
    for i, (table_name, cluster_id) in enumerate(zip(table_names, labels)):
        clusters_dict[cluster_id].append((i, table_name))
    
    # Log each cluster's contents
    for cluster_id in sorted(clusters_dict.keys()):
        tables_in_cluster = clusters_dict[cluster_id]
        
        if cluster_id == -1:
            logger.info(f"\nUNCLUSTERED TABLES ({len(tables_in_cluster)} tables):")
        else:
            logger.info(f"\nCLUSTER {cluster_id} ({len(tables_in_cluster)} tables):")
        
        for idx, table_name in tables_in_cluster:
            logger.info(f"  [{idx:3d}] {table_name}")
    
    # Log multi-label information if available (RAPTOR-specific)
    if multi_labels is not None:
        logger.info("-" * 80)
        logger.info("MULTI-LABEL CLUSTER ASSIGNMENTS (RAPTOR):")
        
        multi_cluster_tables = []
        for i, (table_name, multi_label) in enumerate(zip(table_names, multi_labels)):
            if len(multi_label) > 1:
                multi_cluster_tables.append((i, table_name, multi_label))
        
        if multi_cluster_tables:
            logger.info(f"Tables belonging to multiple clusters: {len(multi_cluster_tables)}")
            for idx, table_name, clusters in multi_cluster_tables:
                cluster_list = ', '.join(map(str, clusters))
                logger.info(f"  [{idx:3d}] {table_name} -> Clusters: [{cluster_list}]")
        else:
            logger.info("No tables belong to multiple clusters")
    
    # Log hierarchical structure information
    if linkage_matrix is not None and len(linkage_matrix) > 0:
        logger.info("-" * 80)
        logger.info("HIERARCHICAL STRUCTURE INFORMATION:")
        logger.info(f"Linkage Matrix Shape: {linkage_matrix.shape}")
        logger.info(f"Hierarchical Levels: {len(linkage_matrix)}")
        
        if len(linkage_matrix) > 0:
            # Log some statistics about the dendrogram
            distances = linkage_matrix[:, 2]  # Third column contains distances
            logger.info(f"Average Merge Distance: {np.mean(distances):.4f}")
            logger.info(f"Max Merge Distance: {np.max(distances):.4f}")
            logger.info(f"Min Merge Distance: {np.min(distances):.4f}")
    
    # Cluster quality metrics
    logger.info("-" * 80)
    logger.info("CLUSTER QUALITY METRICS:")
    
    # Calculate cluster size statistics
    cluster_sizes = [len(clusters_dict[cid]) for cid in unique_labels if cid != -1]
    if cluster_sizes:
        logger.info(f"Average Cluster Size: {np.mean(cluster_sizes):.2f}")
        logger.info(f"Largest Cluster Size: {max(cluster_sizes)}")
        logger.info(f"Smallest Cluster Size: {min(cluster_sizes)}")
        logger.info(f"Cluster Size Std Dev: {np.std(cluster_sizes):.2f}")
    
    # Balance metric
    if num_clusters > 1:
        balance = np.std(cluster_sizes) / np.mean(cluster_sizes) if cluster_sizes else 0
        logger.info(f"Cluster Balance (lower is more balanced): {balance:.3f}")
    
    logger.info("=" * 80)
    logger.info("END OF CLUSTER ANALYSIS REPORT")
    logger.info("=" * 80)

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
    clusteror = HDBScan()
    
    encodings = encoder.encode(texts)
    labels = clusteror.cluster(encodings)

    log_clusters(table_names=table_names, labels=labels)

if __name__ == "__main__":
    main()