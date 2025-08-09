from Database import Database
from openai import OpenAI
from app.compartmentalization.clustering.clusterors.HDBScan import HDBScan
from app.compartmentalization.clustering.encoders.sentence_transformer import SentenceTransformerEncoder
import os

hdbscan = HDBScan()
sentence_transformer = SentenceTransformerEncoder()

database = Database()
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
