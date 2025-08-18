import re 
import traceback

PROBLEM_METADATA_TEMPLATE = """**INPUT ANALYSIS TASK**
Extract metadata from the provided input with absolute precision. Do not add, infer, or modify any information.

**INPUT CONTENT**
{questions_text}
{attachments_text}

**SECTION A: QUESTIONS EXTRACTION**
1. Identify EVERY analytical question/problem
2. PRESERVE EXACT wording from input
3. For ambiguous questions, formulate CLEAR descriptions maintaining original intent
4. Include ALL question-related text including calculation specifics

**SECTION B: DATA SOURCES & STRUCTURE**
1. Identify ALL data locations (DBs, APIs, files, URLs)
2. Specify EXACT access methods (SQL queries, API endpoints, file paths)
3. Extract STRUCTURAL DETAILS:
   - File paths/patterns (use wildcards where specified)
   - Data formats (CSV, Parquet, JSON, etc.)
   - Schema details (tables, columns, data types)
   - Sample data representations
   - Partitioning/organization schemes
4. Include credentials ONLY if explicitly provided in input
5. If database query present:
   - Explicitly state "Public dataset - no credentials required" if no credentials in input
   - Include ALL credentials if present in input

**SECTION C: OUTPUT REQUIREMENTS**
1. Extract EVERY formatting instruction
2. Capture ALL precision/unit requirements
3. Specify encoding requirements (base64, etc.)
4. Note ALL constraints (length limits, file formats)
5. Include output schemas or example outputs
6. If no schema provided, create one based EXCLUSIVELY on input

**CRITICAL RULES**
- COMBINE information from text and images WITHOUT ADDITION
- REFERENCE files by exact name when mentioned
- PRESERVE technical details verbatim
- ALL input details MUST appear in output
- DO NOT INFER missing information"""

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

FILE_LOADING_SCRIPT_TEMPLATE = """
Examine the below data sources and identify the files that can be loaded from a filepath.
Write a script that loads each file into one or more pandas dataframes and creates 'df_list'.

INPUT:
{data_source_text}

Instructions:
- For each attached file, determine how to load it into **one or more** pandas DataFrames based on the file extension.
- Supported formats: CSV, TSV, XLS/XLSX (multi-sheet), JSON, Parquet, and ZIP, PDF etc. (may contain multiple data files).
- If a file contains **multiple tables** (e.g. multi-sheet Excel, multi-query SQL, or ZIP with multiple CSVs), include **all** tables as separate DataFrames in the final list.
- Use `py-tabula` for extracting tables from PDFs (if required).
- Return a single Python script that:
    - Imports any needed modules.
    - Loads all DataFrames into a list called `files_dfs`.

You can read each file from: 'request_data/{request_id}/<filename>'

Example:
```python
files_dfs = [
    pd.read_csv("request_data/{request_id}/file1.csv"),
    pd.read_excel("request_data/{request_id}/file2.xlsx")
]

IMPORTANT POINTS
Do not use any try-except blocks in the script.
"""

FILE_LOADING_SCRIPT_SCHEMA = {
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
{script}

ERROR TRACEBACK:
{traceback_text}

TRIED FIXES:
{fix_history_text}

Only fix the error causing parts of the script. Do not touch anything else.

IMPORTANT POINT
{attempt_specific_text}
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
You will be given an INPUT containing data sources and questions.
For EACH question, write a python script with 'find_answer()' function that returns the answer.
---
INPUT:
{data_source_text}

{questions_list_text}

{dfs_text}

KEY INSTRUCTIONS:
 - [VERY IMPORTANT] Use only common data science libraries like numpy, pandas, scipy, matplotlib, seaborn, sklearn etc.
 - Your response should include the scripts in the same order as the questions.
 - Do not use any try-except blocks in the script. 
 - answers that are base64 string MUST BE PREFIXED with 'data:image/<filetype>;base64,'
 - The function should always have the signature 'find_answer()'
 - Import the packages you require in the script.
"""

QUESTION_SCRIPTS_SCHEMA = {
    "type": "object",
    "required": ["scripts"],
    "properties": {
        "scripts": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "script containing the 'find_answer()' function"
            }
        }
    }
}

FIX_QUESTION_SCRIPT_TEMPLATE = """
You will be given the following inputs:
1. Problem metadata containing data sources and questions.
2. A TARGET QUESTION from the questions list.
2. 'find_answer' function for the TARGET QUESTION.
3. A detailed error traceback.
4. Short description of failed fixes that did not work.

Your task is to fix the function based on what it is *trying* to do.
Do not many any other changes.

Here are the inputs:

PROBLEM METADATA:
{data_source_text}

{questions_list_text}

{dfs_text}

TARGET QUESTION
{question_string}

FAILED FUNCTION DEFINITION:
{script}

ERROR TRACEBACK:
{traceback}

FAILED FIXES: 
{fix_history}

KEY INSTRUCTIONS:
- Do not use any try-except blocks in the function definition
- THE 'find_answer' must be present and only return the answer to: '{question_string}'
- Only include the function definition (as string) in your response.
- [VERY IMPORTANT] Use only well known data science libraries like numpy, pandas, scipy, matplotlib, seaborn, sklearn etc.
"""

FIX_QUESTION_SCRIPT_SCHEMA = {
    "type": "object",
    "required": ["fixed_function", "fix_description"],
    "properties": {
        "fixed_function": {
            "type": "string",
            "description": "the fixed function definition"
        },
        "fix_description": {
            "type": "string",
            "description": "compact description of the tried fix in 10-15 words"
        }
    }
}


OUTPUT_SCRIPT_TEMPLATE = """
You will be given the following inputs related to a data analysis problem:
- Questions list
- Snippets of the calculated answers 
- Output format instructions 

Here are the inputs:

QUESTIONS LIST:
{questions_list_text}

CALCULATED ANSWERS LIST:
```python
[
{answers_list_text}
]
```
OUTPUT FORMAT INSTRUCTIONS:
{output_format_text}

YOUR TASK 
Write a python script with 'create_output(answers: list)' function that accepts the answers list 
and returns the answers in the expected format.

YOU MUST DO THE FOLLOWING:
- CLEARLY EXAMINE THE CALCULATED ANSWERS LIST.
- CONVERT EACH ANSWER TO SPECIFIED DATATYPE IN THE OUTPUT FORMAT INSTRUCTIONS.
- [VERY IMPORTANT] CREATE A VALID DUMMY ANSWER FOR THE QUESTIONS WHERE THE ANSWER IS None. 
- MAKE THE DUMMY ANSWER FOR BASE64 STRINGS AS 'data:image/png;base64,iV...(truncated)
- THE OUTPUT MUST FOLLOW THE EXPECTED FORMAT AND SHOULD BE JSON SERIALIZABLE.
- ONLY RETURN THE SCRIPT WITH IMPORTS  OR 'create_output(answers:list)', NO EXPLANATIONS, NO EXAMPLES NOTHING ELSE.
"""

OUTPUT_SCRIPT_SCHEMA = {
    "type": "object",
    "required": ["script"],
    "properties": {
        "script": {
            "type": "string",
            "description": "python script containing create_output(answers: list)"
        }, 
    }
}

FIX_OUTPUT_SCRIPT_TEMPLATE = """
You will be given the following inputs related to a data analysis problem:
- Questions list
- Snippets of the calculated answers 
- Output format instructions 
- Script containing 'create_output(answers)' that is supposed to assemble the answers in the expected format
- Error occured in the script
- Previously tried fixes that did not work.

Here are the inputs:

QUESTIONS LIST:
{questions_list_text}

CALCULATED ANSWERS LIST:
```python
[
{answers_list_text}
]
```
OUTPUT FORMAT INSTRUCTIONS:
{output_format_text}

SCRIPT
{script}

TRACEBACK 
{traceback_text}

Return the raw code snippet of the fixed script such that the output is properly assembled.
"""

FIX_OUTPUT_SCRIPT_SCHEMA = {
    "type": "object",
    "required": ["fixed_script", "fix_description"],
    "properties": {
        "fixed_script": {
            "type": "string",
            "description": "python script containing the fixed create_output(answers: list)"
        }, 
        "fix_description": {
            "type": "string",
            "description": "description of the applied fix in 10-15 words"
        }
    }
}


DFS_TEXT_TEMPLATE = """
Required data has been loaded as dataframes in 'dfs' list.
You MUST access each df in your script using 'dfs[<table_index>]'.

DATA SNIPPETS:
{all_snippets_text}

DFs INSTRUCTIONS
- DO NOT USE ANY OTHER WAY TO ACCESS DATA PRESENT IN SNIPPETS, ALWAYS USE 'dfs[<table_index>]'
- [VERY IMPORTANT] Choose the SMALLEST SUBSET of elements from 'dfs' to find the answers to ALL of the questions.
- Don't join or concatenate the individual dfs UNLESS required for the question.
- [VERY IMPORTANT] PREPROCESS NUMERICAL COLUMNS USING REGEX
"""

def create_questions_list_text(questions_list):
    text = ""
    for question in questions_list:
        text += f"-{question}\n"
    return text 


def create_metadata_text(metadata_dict):
    temp = """
PROBLEM METADATA: 

A. Data Source Information
{data_source_text}

B. Analytical Questions
{questions_list_text}

C. Output Instructions & Format
{output_format_text}
    """
    text = temp.format(
        data_source_text = metadata_dict["data_source_text"],
        questions_list_text = create_questions_list_text(metadata_dict["questions"]),
        output_format_text = metadata_dict["output_format_text"]
    )
    return text 

def create_attachments_text(p):
    text = ""
    if len(p.images) > 0:
        text += f"You have been given {len(p.images)} images with the problem." 
    if len(p.filenames) > 0:
        filepaths = [f"request_data/{p.request_id}/{fn}" for fn in p.filenames]
        text += f"\nAttached files can be read locally at: \n{'\n'.join(filepaths)}"
    return text


def create_traceback_text(e):
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    full_traceback = ''.join(tb_lines)
    sanitized_traceback = re.sub(r'File ".*?",', 'File "<redacted>",', full_traceback)
    return sanitized_traceback

def create_dfs_text(dfs) -> str:
    if not dfs:
        return ''
    
    from prompt_util import DFS_TEXT_TEMPLATE

    all_snippets_text = ""
    for i, df in enumerate(dfs):
        # truncate cell values 
        df = df.map(lambda x: (str(x)[:30] + '...') if isinstance(x, str) and len(x) > 30 else x)

        # create markdown snippet 
        snippet_text = df.sample(min(3, df.shape[0])).to_markdown()
        all_snippets_text += f"\ndfs[{i}] snippet:\n{snippet_text}\n"

    dfs_text = DFS_TEXT_TEMPLATE.format(all_snippets_text=all_snippets_text)
    return dfs_text 

def create_answers_list_text(answers):
    fmtd_answers = []
    for ans in answers:
        ans = str(ans)
        ans = f"{ans[:20]}...(truncated)" if len(ans) > 20 else ans 
        fmtd_answers.append(ans)
    return ",\n".join(fmtd_answers)
