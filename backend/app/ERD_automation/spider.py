import os
import csv
import heapq
from collections import defaultdict
import pandas as pd
from ..helper import database
import json

def load_dataFrames():
    """
    Load all CSV files from the given directory and extract column data.
    
    Args:
        directory_path (str): Path to directory containing CSV files
        
    Returns:
        dict: Dictionary mapping 'table.column' to sorted unique values
    """

    column_dict = {}
    
    table_names = database.r.keys('*')
    for table_name in table_names:

        list_items = database.r.lrange(table_name, 0,-1)
        rows_data = []
        for i in range(len(list_items)):
            row_data=json.loads(list_items[i])
            rows_data.append(row_data)

        # Create DataFrame for analysis
        df = pd.DataFrame(rows_data)

        # Process each column
        for column in df.columns:
            column_key = f"{table_name}.{column}"
            # Get unique values, remove NaN, and sort
            unique_values = df[column].dropna().unique()
            sorted_values = sorted(unique_values)
            
            column_dict[column_key] = sorted_values
            print(f"Loaded {len(sorted_values)} unique values from {column_key}")
    
    return column_dict

def initialize_inclusion_dict(column_dict):
    """
    Initialize inclusion dictionary where each column can potentially 
    be included in all other columns.
    
    Args:
        column_dict (dict): Dictionary of column names to their values
        
    Returns:
        dict: Initial inclusion dictionary
    """
    columns = list(column_dict.keys())
    inclusion_dict = {}
    
    for col in columns:
        # Initially, each column could be included in all other columns
        inclusion_dict[col] = [other_col for other_col in columns if other_col != col]
    
    return inclusion_dict



def spider_algorithm(column_dict):
    """
    Spider algorithm implementation using min-heap to find inclusion dependencies.
    
    Args:
        column_dict (dict): Dictionary mapping column names to sorted unique values
        
    Returns:
        dict: Final inclusion dictionary showing dependencies
    """
    # Initialize inclusion dictionary
    inclusion_dict = initialize_inclusion_dict(column_dict)
    
    # Initialize min heap with all values from all columns
    min_heap = []
    
    print("Building heap...")
    for column in column_dict:
        vals = column_dict[column]
        for val in vals:
            tup = (str(val), column)
            heapq.heappush(min_heap, tup)
    
    print(f"Heap initialized with {len(min_heap)} elements")
    
    # Process heap
    iteration = 0
    while min_heap:
        iteration += 1
        if iteration % 1000 == 0:
            print(f"Processing iteration {iteration}, heap size: {len(min_heap)}")
        
        # Get the smallest value in the heap
        att = []
        current_smallest, var = heapq.heappop(min_heap)
        att.append(var)
        
        # Pop all elements where values are equal to current smallest
        while min_heap and min_heap[0][0] == current_smallest:
            next_var = heapq.heappop(min_heap)[-1]
            att.append(next_var)
        
        # Update inclusion_dict
        # For each attribute in att, it can only be included in other attributes in att
        for a in att:
            if a in inclusion_dict:
                inclusion_dict[a] = list(set(inclusion_dict[a]).intersection(att))
    
    return inclusion_dict

def filter_inclusion_dependencies(inclusion_dict):
    """
    Filter inclusion dependencies to remove self-references and empty lists.
    
    Args:
        inclusion_dict (dict): Raw inclusion dictionary
        
    Returns:
        dict: Filtered inclusion dictionary
    """
    filtered_dict = {}
    
    for dependent, references in inclusion_dict.items():
        # Remove self-references and filter non-empty lists
        filtered_references = [ref for ref in references if ref != dependent]
        if filtered_references:
            filtered_dict[dependent] = filtered_references
    
    return filtered_dict

def find_inclusion_dependencies():
    """
    Main function to find inclusion dependencies from CSV files.
    
    Args:
        directory_path (str): Path to directory containing CSV files
        output_file (str): Path to output file (default: "inclusion_dependencies.txt")
    """
    column_dict = load_dataFrames()
    
    if not column_dict:
        print("No data loaded. Exiting.")
        return
    
    print(f"Loaded {len(column_dict)} columns from CSV files")
    
    # Run Spider algorithm
    inclusion_dict = spider_algorithm(column_dict)
    
    # Filter results
    filtered_dict = filter_inclusion_dependencies(inclusion_dict)
    
    # store inds as a list of tuples in database 
    for dependent in sorted(inclusion_dict.keys()):
            references = inclusion_dict[dependent]
            for reference in sorted(references):
                database.inclusion_dependencies.append((reference,dependent))
    database.filtered = database.inclusion_dependencies
    #     
    print(f"Found {sum(len(refs) for refs in filtered_dict.values())} inclusion dependencies")
   