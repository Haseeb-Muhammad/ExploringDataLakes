from app.compartmentalization.clustering.clusteror.clusteror import Clusteror
import hdbscan
import numpy as np

class HDBScan(Clusteror):

    def __init__(self, min_cluster_size:int=2):
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')

    def cluster(self, embeddings:np.ndarray):
        """
        Perform clustering on the input embeddings.

        Args:
            embeddings (np.ndarray): embeddings of shape [number of tables, embedding dimentions]

        Returns:
            labels (dict{int : List[int]}): Cluster labels for each input point.
        """
        self.clusterer.fit(embeddings)
        return self.clusterer.labels_


