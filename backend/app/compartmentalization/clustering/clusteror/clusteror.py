from abc import ABC, abstractmethod
import numpy as np

def Clusteror(ABC):

    @abstractmethod
    def cluster(self, embeddings:np.ndarray):
        """
        Perform clustering on the input embeddings.

        Args:
            embeddings (np.ndarray): embeddings of shape [number of tables, embedding dimentions]

        Returns:
            labels (List[int]): Cluster labels for each input point.
        """
        pass
