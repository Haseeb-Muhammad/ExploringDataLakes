from .attribute import Attribute
from ..helper import database
import json
import pandas as pd
from .ind import IND

def get_INDs():
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

def prefiltering():
    """
    Filter inclusion dependencies based on primary key and null value criteria.
    
    Parameters
    ----------
    inds : list of IND
        List of inclusion dependency objects to filter
        
    Returns
    -------
    list of IND
        Filtered list of inclusion dependencies that meet criteria:
        - Reference attribute is a primary key
        - Neither dependent nor reference is all null values
        
    Notes
    -----
    Uses global pk_table dictionary for primary key lookup
    """
    inds = get_INDs()

    pruned_inds = []
    for ind in inds:
        #Checking if reference variable is a primary key
        is_pk = False
        if database.primary_keys[ind.reference.table_name][0] == ind.reference.fullName:
            is_pk=True

        #Checking if either all of the dependent or reference attribute is null
        dependent_all_null = True
        reference_all_null = True
        for value in ind.reference.values:
            if value != "nan":
                reference_all_null = False
        for value in ind.dependent.values:
            if value !="nan":
                dependent_all_null = False
        
        if is_pk and (not reference_all_null) and (not dependent_all_null):
            pruned_inds.append(ind)
            
    return pruned_inds
