from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict
import numpy as np
import umap
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from app.compartmentalization.clustering.clusteror.clusteror import Clusteror

class RAPTORClusteror(Clusteror):
    """
    RAPTOR-style clustering implementation that uses UMAP for dimensionality reduction
    followed by Gaussian Mixture Model clustering with probability thresholding.
    """
    
    def __init__(
        self,
        dim: int = 10,
        threshold: float = 0.1,
        max_clusters: int = 50,
        random_seed: int = 224,
        umap_metric: str = "cosine",
        n_neighbors: Optional[int] = None
    ):
        """
        Initialize the RAPTOR clustering system.
        
        Args:
            dim: Target dimensionality for UMAP reduction
            threshold: Probability threshold for GMM cluster assignment
            max_clusters: Maximum number of clusters to consider
            random_seed: Random seed for reproducibility
            umap_metric: Distance metric for UMAP
            n_neighbors: Number of neighbors for UMAP (auto-calculated if None)
        """
        self.dim = dim
        self.threshold = threshold
        self.max_clusters = max_clusters
        self.random_seed = random_seed
        self.umap_metric = umap_metric
        self.n_neighbors = n_neighbors
        
    def _global_cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform global dimensionality reduction using UMAP.
        """
        n_neighbors = self.n_neighbors
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
            
        return umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=self.dim,
            metric=self.umap_metric,
            random_state=self.random_seed
        ).fit_transform(embeddings)
    
    def _local_cluster_embeddings(
        self, 
        embeddings: np.ndarray, 
        num_neighbors: int = 10
    ) -> np.ndarray:
        """
        Perform local dimensionality reduction using UMAP.
        """
        return umap.UMAP(
            n_neighbors=num_neighbors,
            n_components=self.dim,
            metric=self.umap_metric,
            random_state=self.random_seed
        ).fit_transform(embeddings)
    
    def _get_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """
        Determine optimal number of clusters using BIC with Gaussian Mixture Model.
        """
        max_clusters = min(self.max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=self.random_seed)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
            
        return n_clusters[np.argmin(bics)]
    
    def _gmm_cluster(self, embeddings: np.ndarray):
        """
        Cluster embeddings using Gaussian Mixture Model with probability threshold.
        """
        n_clusters = self._get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=self.random_seed)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > self.threshold)[0] for prob in probs]
        return labels, n_clusters
    
    def _perform_hierarchical_clustering(self, embeddings: np.ndarray) -> List[np.ndarray]:
        """
        Perform the full RAPTOR clustering pipeline: global reduction -> global clustering
        -> local reduction -> local clustering.
        """
        if len(embeddings) <= self.dim + 1:
            # Insufficient data for clustering
            return [np.array([0]) for _ in range(len(embeddings))]

        # Step 1: Global dimensionality reduction
        reduced_embeddings_global = self._global_cluster_embeddings(embeddings)
        
        # Step 2: Global clustering
        global_clusters, n_global_clusters = self._gmm_cluster(reduced_embeddings_global)

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        # Step 3: Local clustering within each global cluster
        for i in range(n_global_clusters):
            # Extract embeddings for current global cluster
            global_cluster_mask = np.array([i in gc for gc in global_clusters])
            global_cluster_embeddings = embeddings[global_cluster_mask]

            if len(global_cluster_embeddings) == 0:
                continue
                
            if len(global_cluster_embeddings) <= self.dim + 1:
                # Small cluster - assign all to same local cluster
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings]
                n_local_clusters = 1
            else:
                # Perform local clustering
                reduced_embeddings_local = self._local_cluster_embeddings(
                    global_cluster_embeddings
                )
                local_clusters, n_local_clusters = self._gmm_cluster(
                    reduced_embeddings_local
                )

            # Assign local cluster IDs
            for j in range(n_local_clusters):
                local_cluster_mask = np.array([j in lc for lc in local_clusters])
                local_cluster_embeddings = global_cluster_embeddings[local_cluster_mask]
                
                # Find original indices of these embeddings
                for local_emb in local_cluster_embeddings:
                    original_indices = np.where(
                        np.all(embeddings == local_emb, axis=1)
                    )[0]
                    for idx in original_indices:
                        all_local_clusters[idx] = np.append(
                            all_local_clusters[idx], j + total_clusters
                        )

            total_clusters += n_local_clusters

        return all_local_clusters
    
    def cluster(
        self, 
        embeddings: np.ndarray, 
        return_hierarchy: bool = False
    ) -> Union[List[int], Dict[str, Union[List[int], np.ndarray]]]:
        """
        Perform RAPTOR-style clustering on embeddings.
        
        Args:
            embeddings: Input embeddings of shape [n_samples, n_features]
            return_hierarchy: If True, return hierarchical clustering information
            
        Returns:
            Cluster labels or dict with labels and hierarchy info
        """
        # Perform hierarchical clustering
        cluster_assignments = self._perform_hierarchical_clustering(embeddings)
        
        # Convert multi-label assignments to single labels for primary clustering
        # Use the first cluster assignment for each point, or -1 if no assignment
        labels = []
        for assignment in cluster_assignments:
            if len(assignment) > 0:
                labels.append(int(assignment[0]))
            else:
                labels.append(-1)
        
        if not return_hierarchy:
            return labels
        
        # For hierarchy information, create a linkage matrix using agglomerative clustering
        # This provides a more traditional hierarchical view
        if len(embeddings) > 1:
            # Reduce dimensionality for linkage computation
            reduced_embeddings = self._global_cluster_embeddings(embeddings)
            linkage_matrix = linkage(reduced_embeddings, method='ward')
        else:
            linkage_matrix = np.array([])
        
        return {
            "labels": labels,
            "linkage_matrix": linkage_matrix,
            "multi_labels": cluster_assignments  # RAPTOR-specific: multiple cluster assignments
        }
