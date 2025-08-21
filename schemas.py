problem_metadata = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string", "description": "question string"},
        },
        "data_source_desc": {
            "type": "string",
            "description": "data sources and structure",
        },
        "output_format_desc": {
            "type": "string",
            "description": "output schema and requirements",
        },
    },
    "required": ["questions", "data_source_desc", "output_format_desc"],
    "additionalProperties": False,
}

files_dfs_script = {
    "type": "object",
    "properties": {
        "script": {
            "type": "string",
            "description": "A complete, valid Python script that loads all data files into a list named 'files_dfs'.",
        },
        "packages": {
            "type": "array",
            "description": "List of pip-installable package names required for the script to run.",
            "items": {
                "type": "string",
                "description": "Package name for installation via pip (e.g., 'pandas', 'numpy').",
            },
        },
    },
    "required": ["script", "packages"],
    "additionalProperties": False,
}

fix_files_dfs_script = {
    "type": "object",
    "properties": {
        "fixed_script": {
            "type": "string",
            "description": "A complete, valid Python script that loads DataFrames into a list called files_dfs.",
        },
        "fix_description": {
            "type": "string",
            "description": "Short description of applied fix in 10-15 words.",
        },
        "packages": {
            "type": "array",
            "description": "List of pip-installable package names required for the fixed script to run.",
            "items": {
                "type": "string",
                "description": "Package name for installation via pip (e.g., 'pandas', 'tabula-py').",
            },
        },
    },
    "required": ["fixed_script", "fix_description", "packages"],
    "additionalProperties": False,
}

webscrape_url = {
    "type": "object",
    "properties": {
        "URLs": {
            "type": "array",
            "description" : "list of URLs that need to be scraped",
            "items": {
                "type": "string",
                "description": "url of the webpage"
            }
        }
    },
    "required": ["URLs"],
    "additionalProperties": False,
}


find_answer_scripts = {
    "type": "object",
    "required": ["scripts", "packages"],
    "properties": {
        "scripts": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "Script containing the 'find_answer' function.",
            },
        },
        "packages": {
            "type": "array",
            "description": "List of pip-installable package names required to run the scripts.",
            "items": {
                "type": "string",
                "description": "Package name for installation via pip (e.g., 'pandas', 'numpy').",
            },
        },
    },
    "additionalProperties": False,
}


fix_find_answer_script = {
    "type": "object",
    "required": ["fixed_script", "fix_description", "packages"],
    "properties": {
        "fixed_script": {
            "type": "string",
            "description": "The fixed function definition as a Python script.",
        },
        "fix_description": {
            "type": "string",
            "description": "Compact description of the tried fix in 10-15 words.",
        },
        "packages": {
            "type": "array",
            "description": "List of pip-installable package names required to run the fixed script.",
            "items": {
                "type": "string",
                "description": "Package name for installation via pip.",
            },
        },
    },
    "additionalProperties": False,
}


output_script = {
    "type": "object",
    "required": ["script", "packages"],
    "properties": {
        "script": {
            "type": "string",
            "description": "Python script containing create_output(answers: list) function.",
        },
        "packages": {
            "type": "array",
            "description": "List of pip-installable package names required to run the script.",
            "items": {
                "type": "string",
                "description": "Package name for installation via pip.",
            },
        },
    },
    "additionalProperties": False,
}


fix_output_script = {
    "type": "object",
    "required": ["fixed_script", "fix_description", "packages"],
    "properties": {
        "fixed_script": {
            "type": "string",
            "description": "Python script containing create_output(answers: list) function.",
        },
        "fix_description": {
            "type": "string",
            "description": "Description of the applied fix in 10-15 words.",
        },
        "packages": {
            "type": "array",
            "description": "List of pip-installable package names required to run the fixed script.",
            "items": {
                "type": "string",
                "description": "Package name for installation via pip.",
            },
        },
    },
    "additionalProperties": False,
}

