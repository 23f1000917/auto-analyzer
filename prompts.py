import re
import traceback
from pandas import DataFrame


def problem_metadata(
    questions_text: str, images: list, filenames: list[str], problem_dir: str
) -> str:
    return f"""
**INPUT ANALYSIS TASK**
Extract metadata from the provided input with absolute precision. Do not add, infer, or modify any information.

**INPUT CONTENT**
{questions_text}

{_describe_attachments(images, filenames, problem_dir)}

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

**RULES YOU MUST FOLLOW**
- COMBINE information from text and images WITHOUT ADDITION
- REFERENCE files by exact name when mentioned
- PRESERVE technical details verbatim
- ALL input details MUST appear in output
- DO NOT INFER missing information
    """.strip()


def files_dfs_script(data_source_desc: str, problem_dir: str) -> str:
    return f"""
You will be given an input related to data sources. 
Write a Python script that loads **all tables** or data structures from each file into pandas DataFrames.

INPUT:
{data_source_desc}

SUPPORTED FILE TYPES AND HOW TO LOAD:

- CSV and TSV: `pd.read_csv()`
- Excel (.xls, .xlsx): `pd.read_excel()`, load **all sheets**
- JSON: `pd.read_json()`
- Parquet: `pd.read_parquet()`
- ZIP: extract all files, load all supported files inside
- PDF: use `tabula.read_pdf(..., pages='all', multiple_tables=True)`
- SQLite database (.sqlite, .db): use `sqlite3` or `sqlalchemy` to list all tables and load each via `pd.read_sql_query()`
- HDF5 (.h5): use `pd.read_hdf()`, load all keys as separate DataFrames
- Other common formats you can handle: include appropriate code to load them as pandas DataFrames if possible
- If a file format is unsupported or unknown, skip it silently

FOR EACH FILE:

- Extract **all tables** or datasets available
- Append each loaded DataFrame to a list named `files_dfs`

OTHER REQUIREMENTS:

- Use the file path: `{problem_dir}/<filename>` for all file loads
- Do **not** use try-except blocks
- Return **only** the final Python script with necessary imports and the `files_dfs` list fully constructed
- The script should be executable and handle all files given

EXAMPLE:

```python
import pandas as pd

files_dfs = [
    pd.read_csv("{problem_dir}/data1.csv"),
    pd.read_excel("{problem_dir}/data2.xlsx", sheet_name="Sheet1"),
    pd.read_excel("{problem_dir}/data2.xlsx", sheet_name="Sheet2")
]
```
Note: Your output will be executed via Python's exec() function. Return only a complete, valid Python script containing all imports and the construction of `files_dfs`. Do not include any explanations, comments, or markdown formatting.
    """.strip()


def fix_files_dfs_script(
    broken_script: str,
    e: Exception,
    tried_fixes: list[str],
    try_except_policy: str,
) -> str:

    return f"""
You must FIX the below script so that it loads the dataframes correctly from the directory into a list named files_dfs.

BROKEN SCRIPT:
{broken_script}

TRACEBACK DESCRIPTION:
{_describe_traceback(e)}

PREVIOUSLY TRIED FIXES:
{"\n".join(f"- {fix}" for fix in tried_fixes) if tried_fixes else "No Fixes Tried"}

TRY EXCEPT POLICY:
{try_except_policy}

Only fix the sections of the script that cause the traceback.
    """.strip()


def webscrape_url(data_source_desc: str) -> str:
    return f"""
You are provided with an INPUT that describes data sources for a data analysis task.

YOUR TASK:
Extract and return the URL of the webpage that needs to be scraped. If no such webpage is described, return `null`.

INSTRUCTIONS:
- Only return a URL if **web scraping** is explicitly mentioned in the input.
- The URL must point to a standard **webpage** suitable for scraping.
- Do **not** return URLs related to:
  - Database connections
  - Direct file downloads (e.g., .csv, .xls, .zip)
  - API endpoints (e.g., ending in `.json`, containing `/api/`, etc.)
- Make sure the URL is valid and well-formed.
- If the input lacks a relevant webpage for scraping, return `null`.

INPUT:
{data_source_desc}
    """.strip()


def find_answer_scripts(
    data_source_desc: str,
    questions: list[str],
    dfs: list[DataFrame],
) -> str:
    find_answer_args = "dfs" if len(dfs) > 0 else ""

    return f"""
You will be given an INPUT containing:
- A description of data sources
- A list of questions

Your task is to write a separate Python script for each question, each defining a function named exactly `find_answer({find_answer_args})` that returns the answer.

---

INPUT:
{data_source_desc}

{"\n".join(f"- {q}" for q in questions)}

{_describe_dfs(dfs)}

INSTRUCTIONS:

- ⚠️ [VERY IMPORTANT] Use only `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, and `sklearn`.
- The output must include one script per question, in the same order as the questions appear.
- Do NOT use `try-except` blocks in any of the scripts.
- Each script must define a function named exactly `find_answer({find_answer_args})` that returns the direct answer to the question.
- Import all necessary packages explicitly within each script.
- Do NOT include explanations, comments, or markdown formatting in the output.
- The `find_answer` function should NOT produce any side effects such as printing or file I/O.
""".strip()


def fix_find_answer_script(
    data_source_desc: str,
    target_question: str,
    broken_script: str,
    dfs: list[DataFrame],
    e: Exception,
    tried_fixes: list[str],
) -> str:

    return f"""
You will be given the following inputs:

1. Problem metadata containing data sources.
2. A TARGET QUESTION.
3. Script containing the 'find_answer' function for the TARGET QUESTION.
4. A detailed error traceback.
5. A short description of previous failed fixes.

Here are the inputs:

PROBLEM METADATA:
{data_source_desc}

{_describe_dfs(dfs)}

TARGET QUESTION:
{target_question}

BROKEN SCRIPT:
```python
{broken_script}

TRACEBACK:
{_describe_traceback(e)}

PREVIOUSLY TRIED FIXES:
{"\n".join(f"- {fix}" for fix in tried_fixes)}
    """.strip()


def output_script(
    questions: list[str], answers: list[str], output_format_desc: str
) -> str:

    return f"""
You are given the following inputs related to a data analysis problem:
- A list of questions
- A list of corresponding calculated answers
- A description of the expected output format

Your task is to generate a Python script that defines the function:

    `def create_output(answers: list) -> Any`

This function should take the list of answers and return a structured output that conforms to the specified format.

HERE ARE THE INPUTS:

QUESTIONS:
{"\n".join(f"- {q}" for q in questions)}

CALCULATED ANSWERS:
```python
{_describe_answers(answers)}
```

EXPECTED OUTPUT FORMAT:
{output_format_desc}

YOUR TASK:

Implement the function `create_output(answers: list)` that transforms the raw answers into the required output structure.

YOU MUST FOLLOW THESE RULES:

- Carefully examine the CALCULATED ANSWERS LIST.

- Convert each answer to the exact data type specified in the OUTPUT FORMAT INSTRUCTIONS.

- For any question where the answer is `None`, generate a valid dummy answer:
  - For base64-encoded image strings, use: 'dummy_base64_string'
  - For other types, use a plausible placeholder that matches the expected format.
- Ensure the returned value is a valid native Python data structure (e.g., dict, list, etc.).
- Add the `data:image/<filetype>;base64,` prefix to base64 strings IF required by the output format  OTHERWISE don't include it.
- DO NOT use json.dumps() or any form of manual JSON serialization.
- The output must be JSON-serializable (i.e., contain only data types that can be converted to JSON).
- Only output the complete Python script containing necessary imports (if any) and the create_output function.
- Do not include any explanations, comments, or example usages—only output the code.
"""


def fix_output_script(
    questions: list[str],
    answers: list,
    output_format_desc: str,
    broken_script: str,
    e: Exception,
) -> str:

    return f"""
You are given the following inputs related to a data analysis task:
- A list of questions
- A list of calculated answer snippets
- Output format instructions
- A Python script that defines 'create_output(answers)' but currently raises an error
- The traceback of the error
- A list of previously attempted (but unsuccessful) fixes

Your task is to return a corrected version of the script such that it works as intended — assembling the output in the specified format without errors.

HERE ARE THE INPUTS:

QUESTIONS:
{"\n".join(f"- {q}" for q in questions)}

CALCULATED ANSWERS LIST:
```python
{_describe_answers(answers)}
```

EXPECTED OUTPUT FORMAT:
{output_format_desc}

BROKEN SCRIPT:
```python
{broken_script}
```

TRACEBACK:
```plaintext
{_describe_traceback(e)}
```

YOUR TASK

- Carefully inspect the script and the traceback to identify and fix the issue.
- Ensure that the corrected script produces the expected structure, using the provided answers list and respecting the output format instructions.
- If any answer is `None`, generate a valid dummy placeholder:
  - For base64-encoded images, use: 'dummy_base64_string'`
  - For other types, use appropriate dummy values.
- The function must return a native Python data structure (e.g., dict, list) — **do not use `json.dumps()`** or other serializers.
- The output must be valid and JSON-serializable.
- Return only the **fixed script** as a raw code snippet — no explanations, comments, or additional text.
    """.strip()


def _describe_attachments(images: list, filenames: list[str], problem_dir: str) -> str:
    desc = ""
    if len(images) > 0:
        desc += f"You have been given {len(images)} images with the problem."
    if len(filenames) > 0:
        filepaths = [f"- {problem_dir}/{fn}" for fn in filenames]
        if desc:
            desc += "\n"
        desc += "Attached files can be read locally at:\n" + "\n".join(filepaths)
    return desc


def _describe_traceback(e: Exception, max_lines=15):
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    full_traceback = "".join(tb_lines)
    sanitized_traceback = re.sub(r'File ".*?",', 'File "<redacted>",', full_traceback)
    # Keep only the last max_lines lines
    tb_split = sanitized_traceback.strip().split("\n")
    shortened = "\n".join(tb_split[-max_lines:])
    return shortened


def _describe_dfs(dfs: list[DataFrame]):
    dfs_desc = f"dfs is a list of dataframes with {len(dfs)} dataframes.\n\n"
    for i, df in enumerate(dfs):
        dfs_desc += f"dfs[{i}] (shape: {df.shape}):\n"
        dfs_desc += (
            df.head(3)
            .map(lambda x: f"{str(x)[:10]}..." if len(str(x)) > 20 else x)
            .to_string(index=False)
        )
        dfs_desc += "\n---\n"

    return f"""
The required data has been loaded into a list of Pandas DataFrames named `dfs`.
Each DataFrame must be accessed using its index: `dfs[<table_index>]`.

DATA SNIPPETS:
{dfs_desc}

INSTRUCTIONS FOR USING DATAFRAMES:

- If all the needed data to solve the problem is present in the snippets, access data **only** via `dfs[<table_index>]`. 
- If the questions require making a database query, you may do so.
- ✅ Always select the **smallest possible subset** of DataFrames necessary to answer **all** the questions.
- 🚫 Do **not** join, merge, or concatenate DataFrames **unless** it is explicitly required to answer a question.
- 🧹 [VERY IMPORTANT] Clean and preprocess **numerical columns** using regular expressions if needed (e.g., remove symbols, convert types).

Adhere strictly to these rules to ensure clean, efficient, and accurate analysis.
    """.strip()


def _describe_answers(
    answers: list, max_items: int = 10, max_item_length: int = 25
) -> str:
    result = []
    for item in answers[:max_items]:
        item_str = str(item)
        if len(item_str) > max_item_length:
            item_str = item_str[: max_item_length - 3] + "..."
        result.append(f"- {item_str}")

    if len(answers) > max_items:
        result.append(f"...and {len(answers) - max_items} more items not shown.")

    return "\n".join(result)

