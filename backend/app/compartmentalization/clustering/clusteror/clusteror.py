from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict
import numpy as np

class Clusteror(ABC):
    @abstractmethod
    def cluster(
        self, 
        embeddings: np.ndarray, 
        return_hierarchy: bool = False
    ) -> Union[List[int], Dict[str, Union[List[int], np.ndarray]]]:
        """
        Perform clustering on the input embeddings.

        Args:
            embeddings (np.ndarray): Embeddings of shape [number of items, embedding dimensions]
            num_clusters (Optional[int]): Desired number of clusters. Optional for hierarchical clustering.
            return_hierarchy (bool): If True, return extra hierarchy info (like linkage matrix) for hierarchical methods.

        Returns:
            - If return_hierarchy is False:
                labels (List[int]): Cluster labels for each input point.
            - If return_hierarchy is True:
                {
                    "labels": List[int],
                    "linkage_matrix": np.ndarray  # only for hierarchical clustering
                }
        """
        pass
