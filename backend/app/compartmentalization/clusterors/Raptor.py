import sys
from pathlib import Path

# Allow running this file directly from anywhere by adding project backend dir to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from typing import List

from app.compartmentalization.clusterors.clusteror import Clusteror
from app.compartmentalization.encoders.encoder import Encoder
from app.compartmentalization.encoders.sentence_transformer import SentenceTransformerEncoder


class Raptor(Clusteror):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder=encoder)
        

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



def main():
    text = [
        "REGION : Stores region identifiers and their descriptions.",
        "CATEGORIES : Stores product categories and their descriptions.",
        "EMPLOYEE_TERRITORIES : Associates employees with the territories they cover.",
        "SUPPLIERS : Stores supplier company details for sourcing products.",
        "US_STATES : Stores US state codes and their regions.",
        "ORDERS : Stores customer orders and shipping information.",
        "ORDER_DETAILS : Stores detailed line items for each order.",
        "TERRITORIES : Stores sales territories and their associated region.",
        "PRODUCTS : Stores product details available for sale.",
        "EMPLOYEES : Stores employee personal and job-related information.",
        "SHIPPERS : Stores shipping companies used for orders.",
        "CUSTOMERS : Stores customer company and contact information."
    ]
    encoder = SentenceTransformerEncoder()
    raptor = Raptor(encoder=encoder)
    print(raptor.cluster(text=text))


if __name__ == "__main__":
    main()