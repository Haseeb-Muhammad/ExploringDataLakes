import sys
from pathlib import Path


# Allow running this file directly from anywhere by adding project backend dir to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from typing import List

from app.compartmentalization.clusterors.clusteror import Clusteror
from app.compartmentalization.encoders.encoder import Encoder
from app.compartmentalization.encoders.sentence_transformer import SentenceTransformerEncoder
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import umap

RANDOM_SEED = 123


# Utility functions from the original code
def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """Perform global dimensionality reduction using UMAP"""
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED
    ).fit_transform(embeddings)

def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """Perform local dimensionality reduction using UMAP"""
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED
    ).fit_transform(embeddings)

def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """Determine optimal number of clusters using BIC with Gaussian Mixture Model"""
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]

def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = RANDOM_SEED):
    """Cluster embeddings using Gaussian Mixture Model based on probability threshold"""
    n_clusters = get_optimal_clusters(embeddings, random_state=random_state)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """Perform hierarchical clustering using global and local clustering"""
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # Global clustering
    global_clusters, n_global_clusters = GMM_cluster(reduced_embeddings_global, threshold)

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        # Extract embeddings belonging to the current global cluster
        global_cluster_mask = np.array([i in gc for gc in global_clusters])
        global_cluster_embeddings_ = embeddings[global_cluster_mask]

        if len(global_cluster_embeddings_) == 0:
            continue
            
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Local dimensionality reduction and clustering
            reduced_embeddings_local = local_cluster_embeddings(global_cluster_embeddings_, dim)
            local_clusters, n_local_clusters = GMM_cluster(reduced_embeddings_local, threshold)

        # Assign local cluster IDs
        global_indices = np.where(global_cluster_mask)[0]
        
        for j in range(n_local_clusters):
            local_cluster_mask = np.array([j in lc for lc in local_clusters])
            local_indices_in_global = np.where(local_cluster_mask)[0]
            
            for local_idx in local_indices_in_global:
                global_idx = global_indices[local_idx]
                all_local_clusters[global_idx] = np.append(
                    all_local_clusters[global_idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters

class RAPTORClusteror(Clusteror):
    """RAPTOR Clustering implementation for database tables"""
    
    def __init__(self, encoder: Encoder, dim: int = 10, threshold: float = 0.1, n_levels: int = 3):
        """
        Initialize RAPTOR Clusteror
        
        Args:
            encoder: Encoder instance for generating embeddings
            dim: Dimensionality for UMAP reduction
            threshold: Probability threshold for GMM clustering
            n_levels: Maximum number of hierarchical levels
        """
        super().__init__(encoder)
        self.dim = dim
        self.threshold = threshold
        self.n_levels = n_levels
    
    def extract_table_names(self, texts: List[str]) -> List[str]:
        """Extract table names from text descriptions"""
        table_names = []
        for text in texts:
            if ':' in text:
                table_name = text.split(':')[0].strip()
                table_names.append(table_name)
            else:
                # Fallback if no colon found
                table_names.append(text.strip())
        return table_names
    
    def embed_and_cluster_texts(self, texts: List[str]) -> pd.DataFrame:
        """Embed texts and perform clustering"""
        # Generate embeddings using the encoder
        text_embeddings = self.encoder.encode(texts)
        text_embeddings_np = np.array(text_embeddings)
        
        # Perform clustering
        cluster_labels = perform_clustering(text_embeddings_np, self.dim, self.threshold)
        
        # Create DataFrame
        df = pd.DataFrame()
        df["text"] = texts
        df["table_name"] = self.extract_table_names(texts)
        df["embd"] = list(text_embeddings_np)
        df["cluster"] = cluster_labels
        
        return df
    
    def expand_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand DataFrame to handle multiple cluster assignments per text"""
        expanded_list = []
        
        for index, row in df.iterrows():
            if len(row["cluster"]) == 0:
                # Handle case where no cluster is assigned
                expanded_list.append({
                    "text": row["text"],
                    "table_name": row["table_name"],
                    "embd": row["embd"],
                    "cluster": -1  # Unassigned cluster
                })
            else:
                for cluster in row["cluster"]:
                    expanded_list.append({
                        "text": row["text"],
                        "table_name": row["table_name"],
                        "embd": row["embd"],
                        "cluster": cluster
                    })
        
        return pd.DataFrame(expanded_list)
    
    def create_summaries(self, expanded_df: pd.DataFrame) -> List[str]:
        """Create summaries for each cluster"""
        all_clusters = expanded_df["cluster"].unique()
        summaries = []
        
        for cluster_id in all_clusters:
            if cluster_id == -1:  # Skip unassigned clusters
                continue
                
            cluster_texts = expanded_df[expanded_df["cluster"] == cluster_id]["text"].tolist()
            
            # Simple summarization - join table descriptions
            summary = f"Cluster {cluster_id} contains tables: " + \
                     "; ".join(cluster_texts)
            summaries.append(summary)
        
        return summaries, [c for c in all_clusters if c != -1]
    
    def cluster_level(self, texts: List[str], level: int) -> Tuple[pd.DataFrame, List[str], List[int]]:
        """Perform clustering for a single level"""
        print(f"Processing level {level} with {len(texts)} texts")
        
        # Embed and cluster
        df_clusters = self.embed_and_cluster_texts(texts)
        
        # Expand clusters for easier processing
        expanded_df = self.expand_clusters(df_clusters)
        
        # Create summaries
        summaries, cluster_ids = self.create_summaries(expanded_df)
        
        print(f"Generated {len(cluster_ids)} clusters at level {level}")
        
        return expanded_df, summaries, cluster_ids
    
    def cluster(self, text: List[str]) -> Dict[int, Dict[int, List[str]]]:
        """
        Main clustering method that returns hierarchical clusters
        
        Returns:
            Dict[level, Dict[cluster_no, List[table_names]]]
        """
        results = {}
        current_texts = text.copy()
        current_table_names = self.extract_table_names(text)
        
        for level in range(1, self.n_levels + 1):
            # Perform clustering for current level
            expanded_df, summaries, cluster_ids = self.cluster_level(current_texts, level)
            
            # Build result structure for this level
            level_clusters = {}
            
            for cluster_id in cluster_ids:
                cluster_tables = expanded_df[expanded_df["cluster"] == cluster_id]["table_name"].tolist()
                level_clusters[int(cluster_id)] = cluster_tables
            
            results[level] = level_clusters
            
            # Check if we should continue to next level
            if len(cluster_ids) <= 1 or level >= self.n_levels:
                break
            
            # Use summaries as input for next level
            current_texts = summaries
            # For next level, we'll track which original tables belong to each summary
            # This is a simplification - in practice you might want more sophisticated tracking
            
        return results

# Example usage and testing
if __name__ == "__main__":
    # Mock encoder for testing
    class MockEncoder(Encoder):
        def encode(self, texts: List[str]) -> np.ndarray:
            # Simple mock encoding - in practice use a real encoder like OpenAI, Sentence Transformers, etc.
            np.random.seed(42)
            return np.random.randn(len(texts), 384)  # 384-dimensional embeddings
    
    # Test data
    text = [
        "REGION : Stores region identifiers and their descriptions.",
        "CATEGORIES : Stores product categories and their descriptions.",
        "EMPLOYEE_TERRITORIES : Associates employees with the territories they cover.",
        "SUPPLIERS : Stores supplier company details for sourcing products.",
        "US_STATES : Stores US state codes and their regions.",
        "ORDERS : Stores customer orders and shipping information.",
        "ORDER_DETAILS : Stores detailed line items for each order.",
        "TERRITORIES : Stores sales territories and their associated region.",
        "PRODUCTS : Stores product details available for sale.",
        "EMPLOYEES : Stores employee personal and job-related information.",
        "SHIPPERS : Stores shipping companies used for orders.",
        "CUSTOMERS : Stores customer company and contact information."
    ]
    
    # Create and test the clusteror
    encoder = SentenceTransformerEncoder()
    clusteror = RAPTORClusteror(encoder, dim=5, threshold=0.1, n_levels=3)
    
    # Perform clustering
    result = clusteror.cluster(text)
    
    # Display results
    print("\n=== RAPTOR Clustering Results ===")
    for level, clusters in result.items():
        print(f"\nLevel {level}:")
        for cluster_no, table_names in clusters.items():
            print(f"  Cluster {cluster_no}: {table_names}")