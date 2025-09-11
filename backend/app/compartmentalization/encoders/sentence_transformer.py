from sentence_transformers import SentenceTransformer
from .encoder import Encoder

import numpy as np

class SentenceTransformerEncoder(Encoder):

    def encode(self, texts: list[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
        """Encode a list of input texts into their corresponding vector embeddings.
        
        Loads the specified sentence transformer model and converts the input texts
        into dense vector representations. The model is loaded fresh on each call.
        
        Args:
            texts (list[str]): A list of strings to be encoded into embeddings.
            model_name (str, optional): The name or path of the sentence transformer
                model to use for encoding. Defaults to 'all-MiniLM-L6-v2'.
        
        Returns:
            np.ndarray: A NumPy array containing the embeddings for each input text,
                where each row represents the embedding vector for the corresponding
                input text.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def __str__(self):
        """Return string representation of the encoder.
        
        Returns:
            str: The name of the sentence transformer model currently used by
                this encoder instance.
        """        
        return f"{self.model_name}"