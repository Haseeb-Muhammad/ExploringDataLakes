from sentence_transformers import SentenceTransformer
from .encoder import Encoder

import numpy as np

class SentenceTransformerEncoder(Encoder):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the SentenceTransformer encoder with the specified model.

        Args:
            model_name (str): The name of the pre-trained SentenceTransformer model to use.
                Defaults to 'all-MiniLM-L6-v2'.

        Attributes:
            model (SentenceTransformer): The loaded SentenceTransformer model instance.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encodes a list of input texts into their corresponding vector embeddings.

        Args:
            texts (list[str]): A list of strings to be encoded.

        Returns:
            np.ndarray: A NumPy array containing the embeddings for each input text.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def __str__(self):
        return f"{self.model_name}"