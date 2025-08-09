from abc import ABC, abstractmethod
import numpy as np

class Encoder(ABC):

    @abstractmethod
    def encode(self, texts:list[str])->np.ndarray:
        """
        Encode the provided texts where 

        Args:
            texts (list[str]): list where each element is a string representing a table description. List of shape [number of tables]

        Returns:
            encodings (np.ndarray): list of shape [number of tables, embedding dimentions]
        """
        pass
