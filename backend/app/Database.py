import pandas as pd
import json

class Database:

    def __init__(self):
        self.db_frames : list(pd.DataFrame) = []
        self.gt:json = {}
