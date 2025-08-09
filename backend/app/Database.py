import pandas as pd
import json

class Database:
    """A class for managing database tables, ground truth data, and descriptions.

    This class provides a structure to store multiple pandas DataFrames representing
    database tables, ground truth (GT) data loaded from JSON, and generated descriptions
    for the database tables.

    Attributes:
        db_frames (dict[str, pd.DataFrame]): 
            A dictionary mapping table names (as strings) to their corresponding pandas DataFrames.
        gt (dict): 
            A dictionary containing ground truth typically loaded from a JSON file.
        db_description (dict): 
            A dictionary containing generated descriptions for the database tables.
    """

    def __init__(self):
        """Initializes the Database object with empty data structures."""
        self.db_frames: dict[str, pd.DataFrame] = {}
        self.gt: dict = {}
        self.db_description: dict = {}
