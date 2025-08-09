import pandas as pd
import json

class Database:

    def __init__(self):
        self.db_frames : dict[str: pd.DataFrame] = {}
        self.gt:json = {}
        self.db_description = {}
