from abc import ABC, abstractmethod
from typing import List, Any, Dict
import numpy as np
from ..encoders.encoder import Encoder

class Clusteror(ABC):

    def __init__(self, encoder:Encoder):
        """
        Initializes the Clusteror with a given encoder.

        Args:
            encoder (Encoder): An encoder instance used for data transformation.
        """
        self.encoder = encoder
        super().__init__()

    @abstractmethod
    def cluster(
        self,
        text : List[str],
    ) -> dict:
        """
        Clusters the given list of text data into groups.

        Args:
            text (List[str]): A list of strings to be clustered.

        Returns:
            Return a dict: {level: {clusterNo: [table names]}} 
        """
        pass
