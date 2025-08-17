import os 
import re
import yaml
import json 
import glob
import traceback
import pandas as pd 
from PIL import Image
from io import BytesIO
from typing import Any
from google import genai
from textwrap import dedent 
from dotenv import load_dotenv 
from PIL.ImageFile import ImageFile
from fastapi import FastAPI, Request
from starlette.datastructures import UploadFile

# ------------------------------------------ FastAPI --------------------------------------------------

# load secrets
load_dotenv(".venv/secrets.env")

# initialize app
app = FastAPI()

@app.post("/analyze")
async def analyze(request: Request):

    # create problem instance 
    problem = await create_problem_instance(request)  

    # use LLM to create metadata_dict then create metadata_text using yaml 
    problem.metadata_dict =  generate_problem_metadata(problem) 
    problem.metadata_text =  yaml_dump(problem.metadata_dict)   

    # use metadata to populate other properties
    problem.questions_list = problem.metadata_dict.get("questions", [])
    problem.questions_list_text = yaml_dump(problem.questions_list)
    problem.data_source_text = problem.metadata_dict.get("data_source_text", "")
    problem.output_format_text = problem.metadata_dict.get("output_format_text", "")

    # load dfs from attached files (if any)
    if len(problem.file_dict) > 0:
        problem.files_dfs = load_files_as_dfs(problem)

    # scrape tables from webpage if needed 
    problem.scraped_dfs = webscrape_tables_if_needed(problem)

    # combine dfs from both sources into dfs list and create dfs_text 
    problem.dfs = problem.files_dfs + problem.scraped_dfs
    problem.dfs_text = create_dfs_text(problem.dfs)

    # find the answers to the questions
    answers = find_question_answers(problem)

    output = generate_output(problem, answers)

    files = glob.glob("request_data/*")
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Failed to delete {f}: {e}")
            
    return output



# --------------------------------------- Main Pipeline --------------------------------------------------

# define the problem class 
class Problem:
    def __init__(self) -> None:
        # The properties suffixed with 'text' are to be used with LLM prompts
        self.questions_text: str = ""                  # 'questions.txt' content      
        self.image_dict : dict[str, ImageFile] = {}    #  attached images 
        self.file_dict: dict[str, BytesIO] = {}        #  other attached files 

        self.metadata_dict: dict = {}                  # contains keys: questions_list, data_source_text, output_format_text
        self.metadata_text: str =  ""                 

        # properties populated using 'metadata_dict'
        self.questions_list: list[str] = []            # list of questions 
        self.questions_list_text: str = ""
        self.data_source_text: str = ""                # data source description 
        self.output_format_text: str = ""              # expected output format description


        # dataframes created from attached files and web scraped tables (can be empty)
        self.scraped_dfs: list[pd.DataFrame] = []      # dfs created from webpage tables
        self.files_dfs: list[pd.DataFrame] = []        # dfs created from attached files 
        self.dfs: list[pd.DataFrame] = []              # dfs = scraped_dfs + files_dfs

        self.dfs_text: str = ""  # snippets of scraped and files dfs 
                    



async def create_problem_instance(request: Request) -> Problem:
    # filter out non upload files 
    uploads = [(n, v) for n, v in (await request.form()).items() if isinstance(v, UploadFile)]
    # print(uploads)

    problem = Problem() # create problem instance

    for name, upload_file in uploads:
        if name == "questions.txt":  # questions.txt content stored in problem.questions_text
            problem.questions_text = (await upload_file.read()).decode("utf-8")

        elif str(upload_file.content_type).startswith("image/"): # images are stored in problem.image_dict
            problem.image_dict[name] = Image.open(BytesIO(await upload_file.read()))

        else: # other files are stored in problem.file_dict
            file_content = await upload_file.read()
            with open(f"request_data/{name}", "wb") as f:
                f.write(file_content)
            problem.file_dict[name] = BytesIO(await upload_file.read())

    return problem



def generate_problem_metadata(p: Problem) -> Any:
    from prompts import PROBLEM_METADATA_TEMPLATE, PROBLEM_METADATA_SCHEMA

    attachments_text = "" # additional text for prompt that gives info about attached images/files  

    images = list(p.image_dict.values()) 
    if len(images) > 0:  
        attachments_text += f"You have been given {len(images)} images with the problem." 

    filenames = list(p.file_dict.keys()) 
    if len(filenames) > 0: # only filenames are included in prompt
        attachments_text += f"Names of files attached with this problem: \n-{'\n'.join(filenames)}"

    prompt_text = PROBLEM_METADATA_TEMPLATE.format(questions_text = p.questions_text, attachments_text = attachments_text)
    response_json = ask_LLM(contents = [prompt_text] + images, response_schema = PROBLEM_METADATA_SCHEMA)
    return response_json



def load_files_as_dfs(p: Problem) -> list[pd.DataFrame]:
    from prompts import FILE_LOADING_PROMPT, FILE_LOADING_SCHEMA

    prompt_text = FILE_LOADING_PROMPT.format(data_source_text = p.data_source_text)
    response_json = ask_LLM(contents=[prompt_text], response_schema=FILE_LOADING_SCHEMA) 
    script = response_json.get("script")
    if not script: return []
    
    files_dfs = run_file_loading_script(script, p.file_dict)
    valid_dfs = [df for df in files_dfs if isinstance(df, pd.DataFrame) and not df.empty]
    return valid_dfs



def run_file_loading_script(script: str, files: dict, max_tries: int = 5) -> list:
    from prompts import FIX_FILE_LOADING_TEMPLATE, FIX_FILE_LOADING_SCHEMA
    fix_history_blocks = []

    for attempt in range(max_tries):
        try:
            locals_dict = {}
            exec(script, {}, locals_dict)
            files_dfs = locals_dict.get("files_dfs", [])
            if isinstance(files_dfs, list):
                return files_dfs
            else:
                print("files_dfs is not a list.")
                return []

        except Exception as e:
            # Sanitize traceback
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            full_traceback = ''.join(tb_lines)
            sanitized_traceback = re.sub(r'File ".*?",', 'File "<redacted>",', full_traceback)

            print(f"[Script attempt {attempt + 1}/{max_tries}] Error:\n{sanitized_traceback}")

            # Trial-specific fix instructions
            if attempt < max_tries - 1:
                trial_instruction = "\nFix the error, but DO NOT use try-except blocks in your fix.\n"
            else:
                trial_instruction = "\nCRITICAL INSTRUCTION: Wrap each file load in try-except and skip failing ones.\n"

            # Format fix history block
            if fix_history_blocks:
                fix_history_str = "\n\nFIX HISTORY:\n" + "\n".join(f"- {desc}" for desc in fix_history_blocks)
            else:
                fix_history_str = ""

            # Fill the template with dynamic context
            fix_prompt = FIX_FILE_LOADING_TEMPLATE.format(
                failed_script=script,
                traceback=sanitized_traceback,
                fix_history=fix_history_str if fix_history_str else '',
                trial_instruction = trial_instruction
            )

            # Call LLM to fix the script
            res = ask_LLM(contents=[fix_prompt], response_schema=FIX_FILE_LOADING_SCHEMA)

            # Get fixed script and fix summary
            script = res.get("fixed_script", "")
            fix_description = res.get("fix_description", "").strip()

            fix_history_blocks.append(f"Attempt {attempt + 1}: {fix_description}")

    print("All attempts to fix the script failed.")
    return []



def webscrape_tables_if_needed(p: Problem) -> list[pd.DataFrame]:
    from prompts import WEBSCRAPE_URL_TEMPLATE, WEBSCRAPE_URL_SCHEMA

    prompt_text = WEBSCRAPE_URL_TEMPLATE.format(data_source_text = p.data_source_text)
    response_json = ask_LLM(contents = [prompt_text], response_schema = WEBSCRAPE_URL_SCHEMA)
    URL = response_json.get("URL")
    if not URL: return []
    
    try: scraped_tables = pd.read_html(URL)
    except ValueError: return []
    
    valid_tables = [ table for table in scraped_tables if not table.empty and 
        any(not str(col).startswith("Unnamed") and not isinstance(col, int) for col in table.columns)]
    return valid_tables


def find_question_answers(p: Problem) -> list:
    from prompts import QUESTION_SCRIPTS_TEMPLATE, QUESTION_SCRIPTS_SCHEMA

    prompt_text = QUESTION_SCRIPTS_TEMPLATE.format(metadata_text = p.metadata_text, dfs_text = p.dfs_text)

    # Ask LLM to generate scripts for each question
    response_json = ask_LLM(contents=[prompt_text],response_schema=QUESTION_SCRIPTS_SCHEMA)
    scripts = response_json.get("scripts", [])
    answers = []

    # Run each script and collect answers
    for qno, script in enumerate(scripts):
        answer = run_question_script(script, p, qno, max_tries=5)
        answers.append(answer)
    return answers

def run_question_script(script: str, p: Problem, qno: int, max_tries: int = 10) -> Any:
    from prompts import FIX_QUESTION_SCRIPT_TEMPLATE, FIX_QUESTION_SCRIPT_SCHEMA
    fix_history_blocks = []
    for attempt in range(max_tries):
        try:
            local_scope = {}
            exec(script, {"dfs": p.dfs}, local_scope)
            return local_scope.get("answer")
        
        except Exception as e:
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            full_traceback = ''.join(tb_lines)
            sanitized_traceback = re.sub(r'File ".*?",', 'File "<redacted>",', full_traceback)
        
            if attempt < max_tries:
                prompt_text = FIX_QUESTION_SCRIPT_TEMPLATE.format(
                    metadata_text = p.metadata_text,
                    dfs_text = p.dfs_text,
                    script = script,
                    traceback = sanitized_traceback,
                    fix_history = '\n-'.join(fix_history_blocks) if fix_history_blocks else 'None',
                    question_string = p.questions_list[qno]
                )
                res = ask_LLM(
                    contents = [prompt_text],
                    response_schema = FIX_QUESTION_SCRIPT_SCHEMA,
                )
                script = res.get("fixed_script", "")
                fix_description = res.get("fix_description", "").strip()
                fix_history_blocks.append(f"Attempt {attempt + 1}: {fix_description}")
            else:
                return None 


def generate_output(p: Problem, answers):
    from prompts import OUTPUT_SCRIPT_TEMPLATE, OUTPUT_SCRIPT_SCHEMA

    prompt_text = OUTPUT_SCRIPT_TEMPLATE.format(
        questions_list_text = p.questions_list_text,
        answers_list_text = ", ".join([(str(a)[:20] + "...(truncated)") if len(str(a)) > 20 else str(a) for a in answers]),
        output_format_text = p.output_format_text
    )
    res = ask_LLM(contents=[prompt_text], response_schema= OUTPUT_SCRIPT_SCHEMA)
    script = res.get("function_definition")
    try:
        local_ns = {}
        exec(script, {}, local_ns)
        create_output_func = local_ns.get("create_output", lambda answers: answers)
        return create_output_func(answers)
    except:
        return ["output generation failed"]



def ask_LLM(contents: list, response_schema: dict) -> Any:
    models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash"
    ]
    client = genai.Client(api_key=os.environ.get("GEMINI_AUTH_TOKEN"))
    # print("="*50)
    # print(contents[0])
    # input("press any key...")
    for model_name in models:
        try:
            response = client.models.generate_content(
                model = model_name,
                contents = contents, 
                config = {
                    "response_mime_type": "application/json",
                    "response_json_schema": response_schema
                }
            )
            if not response.text:
                raise ValueError("'response.txt' is None")
            response_json = json.loads(response.text)
            # print("-"*50)
            # print(yaml_dump(response_json))
            # print("="*50)
            return response_json
        except Exception as e: continue
    raise Exception("all gemini models failed")


def create_dfs_text(dfs) -> str:
    if not dfs:
        return ''
    from prompts import DFS_TEXT_TEMPLATE

    all_snippets_text = ""

    for i, df in enumerate(dfs):
        # truncate cell values 
        df = df.map(lambda x: (str(x)[:30] + '...') if isinstance(x, str) and len(x) > 30 else x)

        # create markdown snippet 
        snippet_text = df.sample(min(3, df.shape[0])).to_markdown()
        all_snippets_text += f"\ndfs[{i}] snippet:\n{snippet_text}\n"

    dfs_text = DFS_TEXT_TEMPLATE.format(all_snippets_text=all_snippets_text)
    return dfs_text 

def yaml_dump(data):
    return yaml.dump(data, sort_keys=False, default_flow_style=False, indent=4)
