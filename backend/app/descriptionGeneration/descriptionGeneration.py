from app.Database import Database
from prompt import DOCUMENTATION_PROMPT_TEMPLATE
from app.helper import llm, database
import json

def extract_schema_from_database(sample_size=3):
    schema_summary = {}
    for table_name, df in database.db_frames.items():
        schema_summary[table_name] = {
            "columns": {
                col: {
                    "dtype": str(df[col].dtype),
                    "samples": df[col].dropna().unique()[:sample_size].tolist()
                }
                for col in df.columns
            }
        }
    return schema_summary

def generate_description():
    database_meta_data = extract_schema_from_database(database)

    formatted_prompt = DOCUMENTATION_PROMPT_TEMPLATE.format(
        METADATA=database_meta_data
    )

    response = llm.responses.create(
        model="o4-mini",
        input=formatted_prompt
    )   

    database.db_description = json.loads(response.output_text)

