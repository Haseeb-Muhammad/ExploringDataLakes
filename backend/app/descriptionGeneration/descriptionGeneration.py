from ..Database import Database
from .prompt import DOCUMENTATION_PROMPT_TEMPLATE
import json
from ..helper import llm, database
import pandas as pd

def extract_schema_from_redis(sample_size=3):
    """Extracts schema and sample data from all tables stored in Redis.

    This method works with the Redis list storage format where each table is stored
    as a Redis list with each row as a JSON string.

    Args:
        sample_size (int, optional): The number of unique, non-null samples to extract per column.
            Defaults to 3.

    Returns:
        dict: A dictionary summarizing the schema and sample data for each table. The structure is:
            {
                "table_name": {
                    "columns": {
                        "column_name": {
                            "dtype": "<data type as string>",
                            "samples": [sample1, sample2, ...]
                        },
                        ...
                    }
                },
                ...
            }
    """
    schema_summary = {}
    
    # Get all keys from Redis (these represent table names)
    table_names = database.r.keys("*")
    
    for table_name in table_names:
        
        # Get all rows from the Redis list
        list_items = database.r.lrange(table_name, 0, -1)
        
        rows_data = []
        for i in range(sample_size):
            row_data=json.loads(list_items[i])
            rows_data.append(row_data)

        # Create DataFrame for analysis
        df = pd.DataFrame(rows_data)
        
        # Extract schema information
        schema_summary[table_name] = {
            "columns": {
                col: {
                    "dtype": str(df[col].dtype),
                    "samples": df[col].dropna().unique().tolist()
                }
                for col in df.columns
            }
        }
    
    return schema_summary


def generate_description():
    """Generates and stores a JSON description of the database schema using an LLM.

    This function extracts the schema and sample data from the database, formats a prompt
    using a documentation template, and sends it to a language model to generate a
    structured JSON description. The resulting description is stored in the database's
    `db_description` attribute.

    Returns:
        None
    """
    database_meta_data = extract_schema_from_redis(sample_size=3)

    formatted_prompt = DOCUMENTATION_PROMPT_TEMPLATE.format(
        metadata=database_meta_data
    )

    response = llm.responses.create(
        model="o4-mini",
        input=formatted_prompt
    )   

    print(f"{response.output_text}")

    database.db_description = json.loads(response.output_text)
