from app.compartmentalization.clustering.clusterors.clusteror import Clusteror
from app.compartmentalization.clustering.encoders.encoder import Encoder
import hdbscan
import numpy as np
from collections import defaultdict
from typing import List

class HDBScan(Clusteror):

    def __init__(self, encoder:Encoder, min_cluster_size:int=2):
        super().__init__(encoder=encoder)
        self.encoder = encoder
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')

    def cluster(self, text : List[str]):

        embeddings = self.encoder.encode(text)
        self.clusterer.fit(embeddings)

        labels = self.clusterer.labels_
        
        label_map = defaultdict(list)
        for text, label in zip(text, labels):
            if label_map.get(label) == None:
                label_map[label] = [text]
            else:
                label_map[label].append(text)

        return {1: label_map}
    
    def __str__(self,):
        return f"HDBScan "


