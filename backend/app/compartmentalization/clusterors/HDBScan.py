from .clusteror import Clusteror
from ..encoders.encoder import Encoder
import hdbscan
import numpy as np
from collections import defaultdict
from typing import List,Dict

class HDBScan(Clusteror):

    def __init__(self, encoder: Encoder):
        """Initializes the HDBScan clusterer.
        
        Args:
            encoder (Encoder): The encoder to use for converting text to embeddings.
        """
        super().__init__(encoder=encoder)
        self.encoder = encoder

    def cluster(self, text: List[str], min_cluster_size: int = 2, metric: str = "euclidean") -> dict:
        """Clusters text items using the HDBScan algorithm.
        
        Takes a list of text items, converts them to embeddings using the configured
        encoder, and applies HDBScan clustering to group similar items together.
        
        Args:
            text (List[str]): List of text items to cluster.
            min_cluster_size (int, optional): Minimum number of samples in a cluster.
                Defaults to 2.
            metric (str, optional): Distance metric to use for clustering. 
                Defaults to "euclidean".
        
        Returns:
            dict: A dictionary containing clustering results in the format:
                {1: {cluster_id: [list_of_text_items]}}, where cluster_id -1 
                represents noise/outlier points.
        """
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)


        embeddings = self.encoder.encode(text)
        clusterer.fit(embeddings)

        labels = clusterer.labels_
        
        label_map = defaultdict(list)
        for text_item, label in zip(text, labels):
            # Convert numpy types to regular Python types for JSON serialization
            label = int(label) if hasattr(label, 'item') else label
            if label_map.get(label) is None:
                label_map[label] = [text_item]
            else:
                label_map[label].append(text_item)

        # Convert defaultdict to regular dict for JSON serialization
        return {1: dict(label_map)}
    
    def __str__(self):
        """Returns a string representation of the HDBScan clusteror.

        Returns:
            str: The name of the clusteror.
        """
        return f"HDBScan "

