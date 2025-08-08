DOCUMENTATION_PROMPT_TEMPLATE = """You are an expert database analyst. I will give you metadata extracted from a set of CSV files, where:
- Each CSV file represents a table in a database.
- The file name is the table name.
- For each table, I provide column names, detected data types, and a few example values.
- You will write a JSON description of the database structure in the following format:

{
  "tables": {
    "TABLE_NAME": {
      "note": "Overall purpose of the table, inferred from name, columns, and example data.",
      "columns": {
        "COLUMN_NAME": {
          "type": "<data type>",
          "note": "Meaning and role of the column in context."
        }
      }
    }
  }
}

Guidelines:
- Be concise but descriptive.
- Do not invent information not supported by the data.
- Use simple language but keep it professional.
- Ensure JSON is valid and matches the schema exactly.

Here is the CSV metadata:
{METADATA}
"""