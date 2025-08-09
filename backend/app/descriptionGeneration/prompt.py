DOCUMENTATION_PROMPT_TEMPLATE = """You are an expert in database documentation.
I will give you the schema and sample data for several tables extracted from a database.
Your task is to produce a JSON description of the database in the following format:

{
  "tables": {
    "<TABLE_NAME>": {
      "note": "<One sentence describing what this table stores>",
      "columns": {
        "<COLUMN_NAME>": {
          "type": "<text|number|date|boolean|binary>",
          "note": "<One sentence describing what this column stores and its role in the table>"
        }
      }
    }
  }
}

Here is the extracted schema and samples:

{METADATA}

Follow these rules:
1. Use the provided dtype and samples to infer column purpose and type.
2. Always produce output in valid JSON format.
3. Use concise but precise descriptions.
4. If you are unsure, make the most reasonable guess based on column names and samples.

"""