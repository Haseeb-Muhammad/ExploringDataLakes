from ..Database import Database
from .prompt import DOCUMENTATION_PROMPT_TEMPLATE
import json
from ..helper import llm, database

def extract_schema_from_database(sample_size=3):
    """Extracts schema and sample data from all tables in the database.

    Iterates over each DataFrame in the database, collecting the data type and a sample
    of unique, non-null values for each column.

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
    for table_name, df in database.db_frames.items():
        schema_summary[table_name] = {
            "columns": {
                col: {
                    "dtype": str(df[col].dtype),
                    "samples": df[col].dropna().unique()[:int(sample_size)].tolist()
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
    database_meta_data = extract_schema_from_database()

    formatted_prompt = DOCUMENTATION_PROMPT_TEMPLATE.format(
        metadata=database_meta_data
    )

    response = llm.responses.create(
        model="o4-mini",
        input=formatted_prompt
    )   

    database.db_description = json.loads(response.output_text)
