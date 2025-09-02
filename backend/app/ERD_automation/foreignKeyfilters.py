from .attribute import Attribute
from ..helper import database
import json
import pandas as pd
from .ind import IND

def get_INDs():
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

def update_databae_filtered_inds(inds: list[IND]):
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

def primary_key_check():
    """
    Filters and retains only those INDs (inferred or identified dependencies) where the reference variable is a primary key.

    This function retrieves a list of INDs, checks if the reference variable in each IND is a primary key in the database,
    and appends those INDs to a pruned list if the condition is met. The filtered list is then used to update the database's
    filtered INDs.

    Returns:
        None
    """
    inds = get_INDs()
    pruned_inds = []
    for ind in inds:
        #Checking if reference variable is a primary key
        is_pk = False
        if database.primary_keys[ind.reference.table_name][0] == ind.reference.fullName:
            is_pk=True

        if is_pk:
            pruned_inds.append(ind)

    update_databae_filtered_inds(inds=pruned_inds)

def null_check():
    """
    Filters and prunes INDs (Inclusion Dependencies) where all values in either the reference or dependent columns are "nan".
    This function retrieves a list of INDs, checks each IND to determine if all values in either the reference or dependent columns are "nan".
    If neither column is entirely "nan", the IND is retained. The filtered list of INDs is then updated in the database.
    Returns:
        None
    """
    inds = get_INDs()
    pruned_inds = []
    for ind in inds:
        dependent_all_null = True
        reference_all_null = True
        for value in ind.reference.values:
            if value != "nan":
                reference_all_null = False
        for value in ind.dependent.values:
            if value !="nan":
                dependent_all_null = False
        
        if (not reference_all_null) and (not dependent_all_null):
            pruned_inds.append(ind)
    
    update_databae_filtered_inds(inds=pruned_inds)
    



