from typing import Optional, Literal, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from fuzzywuzzy import fuzz
import os
import logging
import time
from .helper import database
import pandas as pd
from .ERD_automation.ind import IND
import json
from .ERD_automation.attribute import Attribute


# Type alias for similarity methods
SimilarityMethod = Literal["fastText", "sentenceTransformer", "TF_IDF", "partial_ratio"]

class SimilarityCalculator:
    """
    A class to calculate text similarity using multiple methods.

    Parameters
    ----------
    sentence_transformer_model_name : str, optional
        Name of the sentence transformer model to use (default: 'all-MiniLM-L6-v2')
    fast_model_name : str, optional
        Path to the FastText model file (default: local path to cc.en.300.bin)

    Attributes
    ----------
    fast_text_model : fasttext.FastText._FastText
        Loaded FastText model instance
    sentence_transformer_model : SentenceTransformer
        Loaded Sentence Transformer model instance
    tfidf_vectorizer : TfidfVectorizer
        TF-IDF vectorizer instance configured for character n-grams
    """

    def __init__(
        self, 
        sentence_transformer_model_name: str = "all-MiniLM-L6-v2",
        fast_model_name: str = '/home/haseeb/Desktop/EKAI/ERD_automation/codes/my_automation/my_automation/cc.en.300.bin'
    ) -> None:
        self.fast_model_name: str = fast_model_name
        self.sentence_transformer_model_name: str = sentence_transformer_model_name
        
        # Initialize models lazily to avoid loading if not needed
        self._fast_text_model = None
        self._sentence_transformer_model = None
        self.tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    
    @property
    def fast_text_model(self):
        if self._fast_text_model is None:
            try:
                self._fast_text_model = fasttext.load_model(self.fast_model_name)
            except Exception as e:
                logging.warning(f"Failed to load FastText model: {e}")
                raise ValueError(f"FastText model could not be loaded: {e}")
        return self._fast_text_model
    
    @property 
    def sentence_transformer_model(self):
        if self._sentence_transformer_model is None:
            try:
                self._sentence_transformer_model = SentenceTransformer(self.sentence_transformer_model_name)
            except Exception as e:
                logging.warning(f"Failed to load Sentence Transformer model: {e}")
                raise ValueError(f"Sentence Transformer model could not be loaded: {e}")
        return self._sentence_transformer_model
        
    def fastTextModelSimilarity(self, dependent: str, reference: str) -> np.ndarray:
        """
        Calculate similarity between texts using FastText embeddings.

        Parameters
        ----------
        dependent : str
            First text for comparison
        reference : str
            Second text for comparison

        Returns
        -------
        numpy.ndarray
            Cosine similarity score between the text embeddings
        """
        emb1: np.ndarray = self.fast_text_model.get_word_vector(dependent)
        emb2: np.ndarray = self.fast_text_model.get_word_vector(reference)

        emb1_expanded: np.ndarray = np.expand_dims(emb1, 0)
        emb2_expanded: np.ndarray = np.expand_dims(emb2, 0)
        
        similarity: np.ndarray = cosine_similarity(emb1_expanded, emb2_expanded)

        return similarity
    
    def get_INDs(self,):
        """Constructs IND (Inclusion Dependency) objects for filtered column pairs.

        This function retrieves all table names from the Redis database, loads their rows,
        and creates Attribute objects for each column. It then constructs IND objects for
        each filtered inclusion dependency pair found in `database.filtered`, using the
        corresponding Attribute objects as dependent and reference attributes.

        Returns:
            list: A list of IND objects, each representing an inclusion dependency between
                a dependent and a reference attribute.
        """
        table_names = database.r.keys('*')
        attributes = {}

        for table_name in table_names:
            list_items = database.r.lrange(table_name, 0, -1)
            if not list_items:
                continue

            # Use list comprehension for faster row parsing
            rows_data = [json.loads(item) for item in list_items]
            if not rows_data:
                continue

            df = pd.DataFrame(rows_data)

            # Use enumerate and vectorized operations
            for i, column in enumerate(df.columns):
                # Dropna and convert to string in one go, avoid .tolist() if possible
                non_null_values = df[column].dropna().astype(str).values
                if non_null_values.size > 0:
                    attr = Attribute(table_name, column, non_null_values)
                    attr.position = 1 / (i + 1)
                    attributes[f"{table_name}.{column}"] = attr

        inds = []
        for ind in database.filtered:
            inds.append(IND(dependent=attributes[ind[1]], reference=attributes[ind[0]]))

        return inds

    def sentence_transformer_similarity(self, dependent: str, reference: str) -> np.ndarray:
        """
        Calculate similarity between texts using Sentence Transformers.

        Parameters
        ----------
        dependent : str
            First text for comparison
        reference : str
            Second text for comparison

        Returns
        -------
        numpy.ndarray
            Cosine similarity score between the text embeddings
        """
        emb1: np.ndarray = self.sentence_transformer_model.encode(dependent).reshape(1, -1)
        emb2: np.ndarray = self.sentence_transformer_model.encode(reference).reshape(1, -1)
        similarity: np.ndarray = cosine_similarity(emb1, emb2)

        return similarity 

    def TF_IDF_similarity(self, dependent: str, reference: str) -> np.ndarray:
        """
        Calculate similarity between texts using TF-IDF vectorization.

        Parameters
        ----------
        dependent : str
            First text for comparison
        reference : str
            Second text for comparison

        Returns
        -------
        numpy.ndarray
            Cosine similarity score between the TF-IDF vectors
        """
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([dependent] + [reference])
        similarities: np.ndarray = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])

        return similarities
    
    def partial_ratio(self, dependent, reference):
        """
        Calculate partial string ratio similarity between texts using fuzzywuzzy.

        This method finds the best matching substring and calculates similarity,
        which is particularly useful when one string is a substring of another.

        Parameters
        ----------
        dependent : str
            First text for comparison
        reference : str
            Second text for comparison

        Returns
        -------
        float
            Similarity score between 0 and 1, where 1 indicates perfect match
        """
        
        similarity = fuzz.partial_ratio(dependent, reference) / 100
        return similarity
    
    def compare(self, method: SimilarityMethod, dependent: str, reference: str) -> Union[np.ndarray, float]:
        """
        Compare texts using the specified similarity method.

        Parameters
        ----------
        method : SimilarityMethod
            The similarity method to use. Must be one of:
            - "fastText"
            - "sentenceTransformer"
            - "TF_IDF"
            - "partial_ratio"
        dependent : str
            First text for comparison
        reference : str
            Second text for comparison

        Returns
        -------
        Union[numpy.ndarray, float]
            Similarity score between the texts using the specified method

        Raises
        ------
        ValueError
            If the specified method is not supported
        """
        method_map: dict[str, callable] = {
            "fastText": self.fastTextModelSimilarity,
            "sentenceTransformer": self.sentence_transformer_similarity,
            "TF_IDF": self.TF_IDF_similarity,
            "partial_ratio": self.partial_ratio
        }
        
        if method not in method_map:
            raise ValueError(f"Method '{method}' not supported.")
            
        result = method_map[method](dependent=dependent, reference=reference)
        return result.squeeze() if isinstance(result, np.ndarray) else result
    
    @staticmethod
    def update_database_filtered_inds(inds: list[IND]):
        """
        Updates the global database's filtered list with tuples of full names from the provided IND objects.

        Args:
            inds (list[IND]): A list of IND (Inclusion Dependency) objects. Each IND object is expected to have
                'reference' and 'dependent' attributes, both of which have a 'fullName' attribute.

        Side Effects:
            Modifies the global 'database.filtered' list by clearing it and appending tuples containing the
            full names of the reference and dependent columns from each IND in the input list.
        """
        database.filtered = []
        for ind in inds:
            database.filtered.append((ind.reference.fullName, ind.dependent.fullName))
    
    def differentiate_on_similarity(self, method):
        
        inds = self.get_INDs()

        start = time.time()
        logging.info(f"{"-"*50} Evaluating {method} {"-"*50}")
        candidates = []
        for ind in inds:
            dependent_name = ind.dependent.attribute_name 
            reference_name = ind.reference.attribute_name
            similarity = self.compare(method=method, dependent=dependent_name, reference=reference_name)
            if similarity > 0.7:
                candidates.append(ind)
                
            end = time.time()
            exec_time = end-start
            logging.info(f"Execution time: {exec_time}")
        SimilarityCalculator.update_database_filtered_inds(candidates)


if __name__ == "__main__":
    CSV_DIR = "/home/haseeb/Desktop/EKAI/ERD_automation/Dataset/train/northwind"
    SPIDER_IND_RESULT = "/home/haseeb/Desktop/EKAI/ERD_automation/codes/inclusionDependencyWithSpider/spider_results/northwind.txt"
    GT_PATH = "/home/haseeb/Desktop/EKAI/ERD_automation/Dataset/ground_truth/northwind.json"
    METHODS = ["fastText", "sentenceTransformer", "TF_IDF", "partial_ratio"]
    
    # Example usage - uncomment if needed
    # calculator = SimilarityCalculator()
    # similarity = calculator.compare("partial_ratio", "customer_id", "cust_id")
    # print(f"Similarity: {similarity}")