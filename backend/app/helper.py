from .Database import Database
from openai import OpenAI
from .compartmentalization.clusterors.HDBScan import HDBScan
from .compartmentalization.encoders.sentence_transformer import SentenceTransformerEncoder
import os
import redis


sentence_transformer = SentenceTransformerEncoder()
hdbscan = HDBScan(encoder=sentence_transformer)
database = Database()

# Initialize OpenAI client only if API key is available
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    llm = OpenAI(api_key=api_key)
    print(f"LLM is working")
else:
    llm = None
    print("Warning: OPENAI_API_KEY not set. LLM functionality will be disabled.")
