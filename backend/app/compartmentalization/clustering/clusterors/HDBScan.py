from app.compartmentalization.clustering.clusterors.clusteror import Clusteror
from app.compartmentalization.clustering.encoders.encoder import Encoder
import hdbscan
import numpy as np
from collections import defaultdict
from typing import List

class HDBScan(Clusteror):
    """HDBScan clustering class for text data.

    This class applies the HDBSCAN clustering algorithm to text data after encoding
    it into embeddings using a provided encoder. It inherits from the `Clusteror` base class.

    Attributes:
        encoder (Encoder): The encoder used to convert text into embeddings.
        clusterer (hdbscan.HDBSCAN): The HDBSCAN clustering instance.
    """

    def __init__(self, encoder: Encoder, min_cluster_size: int = 2):
        """Initializes the HDBScan clusteror.

        Args:
            encoder (Encoder): An encoder object that provides an `encode` method to convert text to embeddings.
            min_cluster_size (int, optional): The minimum size of clusters; smaller clusters will be considered noise. Defaults to 2.
        """
        super().__init__(encoder=encoder)
        self.encoder = encoder
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')

    def cluster(self, text: List[str]):
        """Clusters a list of text strings using HDBSCAN.

        Encodes the input text into embeddings, fits the HDBSCAN clusterer, and groups
        the original text by their assigned cluster labels.

        Args:
            text (List[str]): A list of text strings to be clustered.

        Returns:
            dict: A dictionary with a single key (1) mapping to another dictionary,
                where each cluster label maps to a list of text strings belonging to that cluster.
        """
        embeddings = self.encoder.encode(text)
        self.clusterer.fit(embeddings)

        labels = self.clusterer.labels_
        
        label_map = defaultdict(list)
        for text_item, label in zip(text, labels):
            if label_map.get(label) is None:
                label_map[label] = [text_item]
            else:
                label_map[label].append(text_item)

        return {1: label_map}
    
    def __str__(self):
        """Returns a string representation of the HDBScan clusteror.

        Returns:
            str: The name of the clusteror.
        """
        return f"HDBScan "

