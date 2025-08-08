from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Tuple
import numpy as np
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from collections import defaultdict, Counter
from app.compartmentalization.clustering.clusterors.clusteror import Clusteror

class RAPTORClusteror(Clusteror):
    """
    RAPTOR-style clustering implementation that creates true hierarchical clusters
    using global clustering followed by recursive local clustering within each global cluster.
    """
    
    def __init__(
        self,
        dim: int = 10,
        threshold: float = 0.1,
        max_clusters: int = 50,
        random_seed: int = 224,
        reduction_method: str = "pca",  # "pca", "tsne", or "lle"
        n_neighbors: Optional[int] = None,
        hierarchy_levels: int = 2  # Number of hierarchical levels
    ):
        """
        Initialize the RAPTOR clustering system.
        
        Args:
            dim: Target dimensionality for reduction
            threshold: Probability threshold for GMM cluster assignment
            max_clusters: Maximum number of clusters to consider
            random_seed: Random seed for reproducibility
            reduction_method: Dimensionality reduction method ("pca", "tsne", or "lle")
            n_neighbors: Number of neighbors for LLE (auto-calculated if None)
            hierarchy_levels: Number of hierarchical levels to create
        """
        self.dim = dim
        self.threshold = threshold
        self.max_clusters = max_clusters
        self.random_seed = random_seed
        self.reduction_method = reduction_method
        self.n_neighbors = n_neighbors
        self.hierarchy_levels = hierarchy_levels
        
    def _get_reducer(self, n_samples: int):
        """Get the appropriate dimensionality reduction method."""
        if self.reduction_method == "pca":
            return PCA(n_components=self.dim, random_state=self.random_seed)
        elif self.reduction_method == "tsne":
            # t-SNE works best with perplexity < n_samples/3
            perplexity = min(30, max(5, n_samples // 3))
            return TSNE(
                n_components=self.dim, 
                random_state=self.random_seed,
                perplexity=perplexity
            )
        elif self.reduction_method == "lle":
            n_neighbors = self.n_neighbors
            if n_neighbors is None:
                n_neighbors = min(int((n_samples - 1) ** 0.5), n_samples - 1)
            n_neighbors = max(1, min(n_neighbors, n_samples - 1))
            return LocallyLinearEmbedding(
                n_neighbors=n_neighbors,
                n_components=self.dim,
                random_state=self.random_seed
            )
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction_method}")
        
    def _reduce_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform dimensionality reduction."""
        reducer = self._get_reducer(len(embeddings))
        return reducer.fit_transform(embeddings)
    
    def _get_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Determine optimal number of clusters using BIC with Gaussian Mixture Model."""
        max_clusters = min(self.max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=self.random_seed)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
            
        return n_clusters[np.argmin(bics)]
    
    def _gmm_cluster(self, embeddings: np.ndarray) -> Tuple[List[int], int]:
        """Cluster embeddings using Gaussian Mixture Model."""
        n_clusters = self._get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=self.random_seed)
        gm.fit(embeddings)
        
        # Get hard assignments instead of soft assignments for cleaner hierarchy
        labels = gm.predict(embeddings).tolist()
        
        return labels, n_clusters
    
    def _create_hierarchical_structure(self, embeddings: np.ndarray) -> Dict:
        """
        Create a true hierarchical clustering structure.
        Returns hierarchy information including cluster tree and assignments.
        """
        if len(embeddings) <= 1:
            return {
                'hierarchy_tree': {},
                'level_assignments': {0: [0] * len(embeddings)},
                'linkage_matrix': np.array([])
            }
        
        # Create linkage matrix for dendrogram visualization
        reduced_embeddings = self._reduce_embeddings(embeddings)
        linkage_matrix = linkage(reduced_embeddings, method='ward')
        
        # Build hierarchical structure
        hierarchy_tree = {}
        level_assignments = {}
        
        # Level 0: Global clustering
        global_reduced = self._reduce_embeddings(embeddings)
        global_labels, n_global = self._gmm_cluster(global_reduced)
        level_assignments[0] = global_labels
        
        # Build hierarchy tree and perform recursive clustering
        hierarchy_tree[0] = {}
        
        for level in range(1, self.hierarchy_levels):
            level_assignments[level] = [-1] * len(embeddings)
            hierarchy_tree[level] = {}
            
            # Get unique clusters from previous level
            prev_clusters = set(level_assignments[level-1])
            if -1 in prev_clusters:
                prev_clusters.remove(-1)
            
            cluster_counter = 0
            
            for parent_cluster in sorted(prev_clusters):
                # Get indices of points in this parent cluster
                parent_indices = [i for i, label in enumerate(level_assignments[level-1]) 
                                if label == parent_cluster]
                
                if len(parent_indices) <= 1:
                    # Single point or empty cluster - assign to same subcluster
                    for idx in parent_indices:
                        level_assignments[level][idx] = cluster_counter
                    hierarchy_tree[level][cluster_counter] = {
                        'parent': parent_cluster,
                        'indices': parent_indices,
                        'size': len(parent_indices)
                    }
                    cluster_counter += 1
                    continue
                
                # Extract embeddings for this parent cluster
                parent_embeddings = embeddings[parent_indices]
                
                if len(parent_embeddings) <= self.dim + 1:
                    # Too few points for meaningful clustering
                    for idx in parent_indices:
                        level_assignments[level][idx] = cluster_counter
                    hierarchy_tree[level][cluster_counter] = {
                        'parent': parent_cluster,
                        'indices': parent_indices,
                        'size': len(parent_indices)
                    }
                    cluster_counter += 1
                else:
                    # Perform local clustering within parent cluster
                    local_reduced = self._reduce_embeddings(parent_embeddings)
                    local_labels, n_local = self._gmm_cluster(local_reduced)
                    
                    # Map local labels to global labels at this level
                    for i, local_label in enumerate(local_labels):
                        global_idx = parent_indices[i]
                        level_assignments[level][global_idx] = cluster_counter + local_label
                    
                    # Update hierarchy tree
                    for local_cluster_id in range(n_local):
                        local_indices = [parent_indices[i] for i, label in enumerate(local_labels) 
                                       if label == local_cluster_id]
                        hierarchy_tree[level][cluster_counter + local_cluster_id] = {
                            'parent': parent_cluster,
                            'indices': local_indices,
                            'size': len(local_indices)
                        }
                    
                    cluster_counter += n_local
        
        return {
            'hierarchy_tree': hierarchy_tree,
            'level_assignments': level_assignments,
            'linkage_matrix': linkage_matrix
        }
    
    def cluster(
        self, 
        text : List[str],
        embeddings: np.ndarray, 
        return_hierarchy: bool = True
    ) -> Union[List[int], Dict[str, Union[List[int], np.ndarray]]]:
        """
        Perform RAPTOR-style hierarchical clustering on embeddings.
        
        Args:
            embeddings: Input embeddings of shape [n_samples, n_features]
            return_hierarchy: If True, return hierarchical clustering information
            
        Returns:
            Cluster labels or dict with labels and hierarchy info
        """
        # Create hierarchical structure
        hierarchy_info = self._create_hierarchical_structure(embeddings)
        
        # Use the deepest level as primary labels
        deepest_level = max(hierarchy_info['level_assignments'].keys())
        labels = hierarchy_info['level_assignments'][deepest_level]
        
        if not return_hierarchy:
            return labels
        
        # Return comprehensive hierarchy information
        return {
            "labels": labels,  # Final level labels
            "linkage_matrix": hierarchy_info['linkage_matrix'],
            "hierarchy_tree": hierarchy_info['hierarchy_tree'],
            "level_assignments": hierarchy_info['level_assignments'],
            "hierarchy_levels": self.hierarchy_levels,
            "multi_labels": None  # Not used in true hierarchical clustering
        }