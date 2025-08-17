PROBLEM_METADATA_TEMPLATE = """
You will be given an INPUT containing a data analysis problem.
You have to create metadata for the problem.

The problem will contain details about:
1. ANALYSIS QUESTIONS
2. DATA SOURCES & STRUCTURE
3. OUTPUT SCHEMA & REQUIREMENTS 

INPUT:
{questions_text}
{attachments_text}

Metadata Extraction Guidelines:

A. QUESTIONS (CRITICAL):
• Extract question strings from the inputs
• Extract EVERY analytical question/problem
• Preserve EXACT wording when unambiguous
• Formulate clear descriptions when needed
• In the question string, include any substring that seems like a part of the question itself or impacts the way the answer is calculated.

B. DATA SOURCES & STRUCTURE (CRITICAL):
• Identify ALL data locations (DBs, APIs, files, URLs, Webpages)
• Capture ACCESS METHODS (queries, endpoints, webscrape, api, direct file download etc.)
• Extract STRUCTURAL DETAILS:
  - File paths/patterns (include wildcards)
  - Data formats (Parquet, CSV, JSON, etc.)
  - Schema details (tables, columns, types)
  - Sample data representations
  - Partitioning/organization schemes
• Include CREDENTIALS/REGIONS if specified
• Include EVERY LITTLE DETAIL, LEAVE NOTHING

C. OUTPUT SCHEMA AND REQUIREMENTS (CRITICAL):
• EVERY formatting instruction (tables, charts, etc.)
• Precision/unit requirements
• Encoding specifications (base64, etc.)
• Constraints (length limits, file formats)
• Output schemas or example outputs 
• If output schema is not present, try to create one 
• Include EVERY LITTLE DETAIL, LEAVE NOTHING

D. CONTEXT HANDLING (CRITICAL):
• COMBINE information from text and images (if present)
• REFERENCE files by name when mentioned
• PRESERVE technical details exactly
• FOR ANY DETAIL IN THE ORIGINAL INPUT, IT MUST BE PRESENT IN YOUR RESPONSE

E. CRITICAL INSTRUCTION
If the data source is a database query, EXPLICITLY mention in the metadata that:
 - Dataset is public and does not require any credentials if no credentials are present
 - If credentials are present, include all that are required to query the database.

"""
PROBLEM_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "question string"
            }
        },
        "data_source_text": {
            "type": "string",
            "description": "data sources and structure"
        },
        "output_format_text": {
            "type": "string",
            "description": "output schema and requirements"
        }
    },
    "required": ["questions", "data_source_text", "output_format_text"],
    "additionalProperties": False
}

FILE_LOADING_PROMPT = """
You will be given an INPUT that mentions data sources for a data analysis problem.
Examine the INPUT and identify which files have been **attached** with the problem.
Do not include data that needs to be extracted from a remote source.

INPUT:
{data_source_text}

Your task is to write a python script that will be executed using `exec()` function.

Instructions:
- For each attached file, determine how to load it into **one or more** pandas DataFrames based on the file extension.
- Supported formats: CSV, TSV, XLS/XLSX (multi-sheet), JSON, Parquet, and ZIP, PDF etc. (may contain multiple data files).
- If a file contains **multiple tables** (e.g. multi-sheet Excel, multi-query SQL, or ZIP with multiple CSVs), include **all** tables as separate DataFrames in the final list.
- Use `py-tabula` for extracting tables from PDFs (if required).
- Return a single Python script that:
    - Imports any needed modules.
    - Loads all DataFrames into a list called `files_dfs`.

All the filepaths are of the form 'request_data/<filename>'

Example:
```python
# example 
files_dfs = [
    pd.read_csv("request_data/file1.csv"),
    pd.read_excel("request_data/file2.xlsx")
]
"""

FILE_LOADING_SCHEMA = {
    "type": "object",
    "properties": {
        "script": {
            "type": "string",
            "description": "A single valid Python script that loads DataFrames into a list called files_dfs."
        }
    },
    "required": ["script"],
    "additionalProperties": False
}

FIX_FILE_LOADING_TEMPLATE = """
You must FIX the below script so that it loads the dataframes correctly from the directory into a list named files_dfs.

FAILED SCRIPT:
{failed_script}

ERROR TRACEBACK:
{traceback}

{fix_history}

Only fix the error causing parts of the script. Do not touch anything else.
You can import any needed packages.

{trial_instruction}
"""

FIX_FILE_LOADING_SCHEMA = {
    "type": "object",
    "properties": {
        "fixed_script": {
            "type": "string",
            "description": "A single valid Python script that loads DataFrames into a list called files_dfs."
        },
        "fix_description": {
            "type": "string",
            "description": "short description of applied fix in 10-15 words"
        }

    },
    "required": ["fixed_script", "fix_description"],
    "additionalProperties": False

}

WEBSCRAPE_URL_TEMPLATE = """
You will be given an INPUT that explains the sources of data for a data analyis problem.

YOUR TASK 
Identify the URL of the webpage that needs to scraped else return null.

KEY INSTRUCTIONS:
- Check if it mentions **web scraping**; if not, make the URL null.
- Otherwise return the url for the webpage that needs to be scraped.

CRITICAL INSTRUCTIONS:
- The URL MUST NOT be related to a database connection, direct file download, or API endpoint.
- It MUST point to a WEBPAGE that needs to be scraped.
- The URL must be a valid webpage URL.

INPUT:
{data_source_text}
"""

WEBSCRAPE_URL_SCHEMA = {
    "type": "object",
    "properties": {
        "URL": {
            "type": ["string", "null"]
        }
    },
    "required": ["URL"],
    "additionalProperties": False
}


QUESTION_SCRIPTS_TEMPLATE = """
You will be given a problem metadata containing data sources, questions and output instructions.
Your task is to write per-question python scripts to find the answers.

CRITICAL INSTRUCTIONS:
 - Send the scripts in order of the questions in the problem statement.
 - The `exec()` function will be used to run your scripts without any human intervention.
 - Do not use any try-except blocks in the scripts. 
 - At the end of each script, store the answer in `answer` variable.
 - THE 'answer' VALUE MUST BE JSON SERIALIZABLE
---
PROBLEM STATEMENT:
{metadata_text}

{dfs_text}

BE CAREFUL, DO NOT USE ANY `TRY-EXCEPT` BLOCKS IN THE SCRIPTS.
"""

QUESTION_SCRIPTS_SCHEMA = {
    "type": "object",
    "required": ["scripts"],
    "properties": {
        "scripts": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "The python script to get the answer"
            }
        }
    }
}

FIX_QUESTION_SCRIPT_TEMPLATE = """
You will be four given inputs:
1. Problem metadata containing questions, output instructions and data sources.
2. The latest version of the python script to solve ONE OF the questions.
3. A detailed traceback of what caused an error in the latest script.
4. Short description of previously tried fixes for all the versions of the script.

Your task is to fix the script and return the fixed version.
The `exec()` function will be used to run your script with no human invervention.
You are allowed to install any packages you desire for the fix using `subprocess.run(["uv", "pip", "install", package_name], check=True)`.
Only focus on fixing the provided script, do not do anything else.

Here are the inputs:

PROBLEM STATEMENT:
{metadata_text}

{dfs_text}

TARGET QUESTION
{question_string}

ERROR CAUSING SCRIPT:
{script}

ERROR TRACEBACK:
{traceback}

PREVIOUSLY TRIED FIXES: 
{fix_history}

CRITICAL INSTRUCTIONS:
- BE CAREFUL, DO NOT USE ANY `TRY-EXCEPT` BLOCKS IN THE SCRIPTS.
- DO NOT SOLVE ANY OTHER QUESTION EXCEPT '{question_string}'
- STORE THE ANSWER TO THE QUESTION IN THE `answer` VARIABLE AT THE END OF THE SCRIPT.
- THE 'answer' VALUE MUST BE JSON SERIALIZABLE
- DO NOT PRINT ANYTHING IN THE SCRIPT
"""

FIX_QUESTION_SCRIPT_SCHEMA = {
    "type": "object",
    "required": ["fixed_script", "fix_description"],
    "properties": {
        "fixed_script": {
            "type": "string",
            "description": "the fixed python script"
        },
        "fix_description": {
            "type": "string",
            "description": "compact description of the tried fix in 10-15 words"
        }
    }
}


OUTPUT_SCRIPT_TEMPLATE = """
You will be given INPUTS related to a data analysis problem.

Here are the inputs:

QUESTIONS LIST:
{questions_list_text}

EXAMPLE ANSWERS LIST:
[{answers_list_text}]

OUTPUT FORMAT INSTRUCTIONS:
{output_format_text}

YOUR TASK 
Write a python function 'create_output(answers: list)' that accepts the answers list 
and returns the answers in the expected format (json serializable).
Do not include anything else in your response. 
ONLY RETURN THE FUNCTION DEFINITION, NO EXPLANATIONS, NO EXAMPLES NOTHING ELSE.
"""

OUTPUT_SCRIPT_SCHEMA = {
    "type": "object",
    "required": ["function_definition"],
    "properties": {
        "function_definition": {
            "type": "string",
            "description": "python function to create the 'output' from 'answers' list"
        }, 
    }
}


DFS_TEXT_TEMPLATE = """
Some of the data required for the questions has already been sourced.
The datasets are stored as pandas dataframes inside the list named 'dfs'.
You can access each df in your script using 'dfs[<table_index>]'.

Here are small snippets for each dataframe:

{all_snippets_text}

CRITICAL INSTRUCTIONS
- Choose the SMALLEST SUBSET of elements from 'dfs' to find the answers to ALL of the questions.
- Don't join or concatenate the individual dfs UNLESS required for the question.
- Don't use `errors='coerce'` in the script, use regex to do the preprocessing for the data.
"""
