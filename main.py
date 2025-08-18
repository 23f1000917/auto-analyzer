import os 
import re
import yaml
import uuid 
import json 
import base64
import logging
import traceback
import numpy as np
import pandas as pd 
import builtins
import shutil  # Added for robust directory removal
from PIL import Image
from io import BytesIO
from typing import Any, Optional
from pathlib import Path
from google import genai
from datetime import datetime
from dotenv import load_dotenv 
from PIL.ImageFile import ImageFile
from fastapi import FastAPI, Request
from starlette.datastructures import UploadFile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# load secrets
load_dotenv(".venv/secrets.env")

# initialize app
app = FastAPI()

@app.post("/analyze")
async def analyze(request: Request):
    logger.info("Starting analysis request")
    start_time = datetime.now()
    problem: Optional[Problem] = None  # Initialize as None for cleanup safety
    try:
        # create problem instance 
        problem = await create_problem_instance(request)  
        logger.info(f"Problem instance created with request_id: {problem.request_id}")
        
        # use LLM to create metadata_dict then create metadata_text using yaml 
        logger.info("Generating problem metadata...")
        
        problem.metadata_dict = generate_problem_metadata(problem) 
        problem.metadata_text = yaml_dump(problem.metadata_dict)
        logger.info(f"Generated metadata:\n{problem.metadata_text}")

        # use metadata to populate other properties
        problem.questions_list = problem.metadata_dict.get("questions", [])
        problem.questions_list_text = yaml_dump(problem.questions_list)
        problem.data_source_text = problem.metadata_dict.get("data_source_text", "")
        problem.output_format_text = problem.metadata_dict.get("output_format_text", "")
        logger.info(f"Questions list: {problem.questions_list_text}")
        logger.info(f"Data source: {problem.data_source_text}")
        logger.info(f"Output format: {problem.output_format_text}")

        # load dfs from attached files (if any)
        if len(problem.filenames) > 0:
            logger.info(f"Loading {len(problem.filenames)} attached files as dataframes")
            problem.files_dfs = load_files_as_dfs(problem)
            logger.info(f"Loaded {len(problem.files_dfs)} dataframes from files")
        else:
            logger.info("No files attached for dataframe loading")

        # scrape tables from webpage if needed 
        logger.info("Checking if web scraping is needed")
        problem.scraped_dfs = webscrape_tables_if_needed(problem)
        if problem.scraped_dfs:
            logger.info(f"Scraped {len(problem.scraped_dfs)} tables from web")
        else:
            logger.info("No tables scraped from web")

        # combine dfs from both sources into dfs list and create dfs_text 
        problem.dfs = problem.files_dfs + problem.scraped_dfs
        problem.dfs_text = create_dfs_text(problem.dfs)
        logger.info(f"Combined {len(problem.dfs)} dataframes available")

        # find the answers to the questions
        logger.info(f"Finding answers to {len(problem.questions_list)} questions")
        answers = find_question_answers(problem)
        logger.info(f"Generated answers: {create_answers_list_text(answers)}")

        output = generate_output(problem, answers)
        logger.info(f"Successfully generated output in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        return output

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": "Analysis failed", "message": str(e)}

    finally:
        # Robust cleanup handling
        if problem is not None and problem.request_data_path.exists():
            try:
                shutil.rmtree(problem.request_data_path, ignore_errors=True)
                logger.info(f"Deleted request directory: {problem.request_data_path}")
            except Exception as e:
                logger.error(f"Failed to delete directory {problem.request_data_path}: {e}")
        elif problem is not None:
            logger.warning(f"Request directory not found: {problem.request_data_path}")


class Problem:
    def __init__(self) -> None:
        logger.info("Initializing new Problem instance")
        self.request_id: str = uuid.uuid4().hex[:8]
        self.request_data_path = Path(f"request_data/{self.request_id}")
        self.request_data_path.mkdir(parents=True, exist_ok=True)

        # The properties suffixed with 'text' are to be used with LLM prompts
        self.questions_text: str = ""                  # 'questions.txt' content      
        self.images: list[ImageFile] = []              # attached images 
        self.filenames: list[str] = []                 # names of attached files 

        self.metadata_dict: dict = {}                  # contains keys: questions_list, data_source_text, output_format_text
        self.metadata_text: str = ""                 

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
    logger.info("Creating problem instance from request")
    try:
        # filter out non upload files 
        uploads = [(n, v) for n, v in (await request.form()).items() if isinstance(v, UploadFile)]
        logger.info(f"Found {len(uploads)} upload files in request")

        problem = Problem() # create problem instance
        found_questions = False

        for name, upload_file in uploads:
            logger.info(f"Processing upload file: {name} (type: {upload_file.content_type})")
            if name == "questions.txt":
                problem.questions_text = (await upload_file.read()).decode("utf-8")
                found_questions = True
                logger.info(f"Loaded questions text: {problem.questions_text[:100]}...")

            elif str(upload_file.content_type).startswith("image/"):
                problem.images.append(Image.open(BytesIO(await upload_file.read())))
                logger.info(f"Added image to images list: {name}")

            else:
                file_content = await upload_file.read()
                file_path = Path(f"request_data/{problem.request_id}/{name}")
                with open(file_path, "wb") as f:
                    f.write(file_content)
                problem.filenames.append(name)
                logger.info(f"Added filename: {name}")

        if not found_questions:
            raise ValueError("No 'questions.txt' file found in the request.")
        
        logger.info(f"Problem instance created with {len(problem.images)} images and {len(problem.filenames)} files")
        return problem

    except Exception as e:
        logger.error(f"Error creating problem instance: {str(e)}")
        raise

def generate_problem_metadata(p: Problem) -> Any:
    from prompts import PROBLEM_METADATA_TEMPLATE, PROBLEM_METADATA_SCHEMA

    logger.info("Generating problem metadata")
    attachments_text = "" # additional text for prompt that gives info about attached images/files  

    if len(p.images) > 0:  
        attachments_text += f"You have been given {len(p.images)} images with the problem." 
        logger.info(f"Included {len(p.images)} images in metadata prompt")

    if len(p.filenames) > 0:
        filepaths = [f"request_data/{p.request_id}/{fn}" for fn in p.filenames]
        attachments_text += f"filepaths of files attached with this problem: \n-{'\n'.join(filepaths)}"
        logger.info(f"Included {len(p.filenames)} filenames in metadata prompt")

    prompt_text = PROBLEM_METADATA_TEMPLATE.format(questions_text = p.questions_text, attachments_text = attachments_text)
    logger.info(f"Metadata prompt text:\n{prompt_text}")

    response_json = ask_LLM(contents = [prompt_text] + p.images, response_schema = PROBLEM_METADATA_SCHEMA)
    logger.info(f"Metadata response:\n{yaml_dump(response_json)}")
    
    return response_json

def load_files_as_dfs(p: Problem) -> list[pd.DataFrame]:
    from prompts import FILE_LOADING_PROMPT, FILE_LOADING_SCHEMA

    logger.info("Loading files as dataframes")
    prompt_text = FILE_LOADING_PROMPT.format(data_source_text = p.data_source_text, request_id=p.request_id)
    logger.info(f"File loading prompt:\n{prompt_text}")

    response_json = ask_LLM(contents=[prompt_text], response_schema=FILE_LOADING_SCHEMA) 
    script = response_json.get("script")
    logger.info(f"Received script for file loading:\n{script}")

    if not script: 
        logger.info("No script returned for file loading")
        return []
    
    files_dfs = run_file_loading_script(script, p.filenames)
    valid_dfs = [df for df in files_dfs if isinstance(df, pd.DataFrame) and not df.empty]
    logger.info(f"Successfully loaded {len(valid_dfs)} valid dataframes from files")
    
    return valid_dfs

def run_file_loading_script(script: str, files: list, max_tries: int = 2) -> list:
    from prompts import FIX_FILE_LOADING_TEMPLATE, FIX_FILE_LOADING_SCHEMA
    fix_history_blocks = []

    logger.info(f"Running file loading script (max attempts: {max_tries})")
    
    for attempt in range(max_tries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_tries} to run file loading script")
            # Create global scope with necessary modules
            env = {
                'pd': pd,
                'np': np,
                '__builtins__': builtins
            }
            exec(script, env)
            files_dfs = env.get("files_dfs", [])
            
            if isinstance(files_dfs, list):
                logger.info(f"Successfully executed file loading script on attempt {attempt + 1}")
                return files_dfs
            else:
                logger.info("files_dfs is not a list")
                return []

        except Exception as e:
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            full_traceback = ''.join(tb_lines)
            sanitized_traceback = re.sub(r'File ".*?",', 'File "<redacted>",', full_traceback)
            logger.info(f"[Script attempt {attempt + 1}/{max_tries}] Error:\n{sanitized_traceback}")

            if attempt < max_tries - 1:
                trial_instruction = "\nFix the error, but DO NOT use try-except blocks in your fix.\n"
            else:
                trial_instruction = "\nCRITICAL INSTRUCTION: Wrap each file load in try-except and skip failing ones.\n"

            if fix_history_blocks:
                fix_history_str = "\n\nFIX HISTORY:\n" + "\n".join(f"- {desc}" for desc in fix_history_blocks)
            else:
                fix_history_str = ""

            fix_prompt = FIX_FILE_LOADING_TEMPLATE.format(
                failed_script=script,
                traceback=sanitized_traceback,
                fix_history=fix_history_str if fix_history_str else '',
                trial_instruction=trial_instruction
            )
            logger.info(f"Fix prompt for attempt {attempt + 1}:\n{fix_prompt}")

            res = ask_LLM(contents=[fix_prompt], response_schema=FIX_FILE_LOADING_SCHEMA)
            script = res.get("fixed_script", "")
            fix_description = res.get("fix_description", "").strip()
            logger.info(f"Received fixed script for attempt {attempt + 1}:\n{script}")
            logger.info(f"Fix description: {fix_description}")

            fix_history_blocks.append(f"Attempt {attempt + 1}: {fix_description}")

    logger.error("All attempts to fix the script failed")
    return []

def webscrape_tables_if_needed(p: Problem) -> list[pd.DataFrame]:
    from prompts import WEBSCRAPE_URL_TEMPLATE, WEBSCRAPE_URL_SCHEMA

    logger.info("Checking if web scraping is needed")
    prompt_text = WEBSCRAPE_URL_TEMPLATE.format(data_source_text = p.data_source_text)
    logger.info(f"Web scrape prompt:\n{prompt_text}")

    response_json = ask_LLM(contents = [prompt_text], response_schema = WEBSCRAPE_URL_SCHEMA)
    URL = response_json.get("URL")
    logger.info(f"Received URL for scraping: {URL}")

    if not URL: 
        logger.info("No URL provided for scraping")
        return []
    
    try: 
        logger.info(f"Attempting to scrape tables from URL: {URL}")
        scraped_tables = pd.read_html(URL)
        logger.info(f"Found {len(scraped_tables)} potential tables")
        
        valid_tables = [table for table in scraped_tables if not table.empty and 
            any(not str(col).startswith("Unnamed") and not isinstance(col, int) for col in table.columns)]
        logger.info(f"Found {len(valid_tables)} valid tables after filtering")
        
        return valid_tables
    except ValueError as e:
        logger.info(f"Failed to scrape tables from URL: {str(e)}")
        return []

def find_question_answers(p: Problem) -> list:
    from prompts import QUESTION_SCRIPTS_TEMPLATE, QUESTION_SCRIPTS_SCHEMA

    logger.info(f"Finding answers to {len(p.questions_list)} questions")
    prompt_text = QUESTION_SCRIPTS_TEMPLATE.format(
        data_source_text = p.data_source_text,
        questions_list_text = p.questions_list_text, 
        dfs_text = p.dfs_text
    )
    logger.info(f"Question answering prompt:\n{prompt_text}")

    response_json = ask_LLM(contents=[prompt_text],response_schema=QUESTION_SCRIPTS_SCHEMA)
    function_defs = response_json.get("function_definitions", [])
    logger.info(f"Received {len(function_defs)} answer scripts for questions")

    answers = []

    for qno, script in enumerate(function_defs):
        logger.info(f"Processing question {qno + 1}/{len(function_defs)}")
        answer = run_question_script(script, p, qno)
        answers.append(json_compatible(answer))
        logger.info(f"Answer for question {qno + 1}: {str(answer)[:100]}...")

    return answers

def run_question_script(function_def: str, p: Problem, qno: int, max_tries: int = 2) -> Any:
    from prompts import FIX_QUESTION_SCRIPT_TEMPLATE, FIX_QUESTION_SCRIPT_SCHEMA
    fix_history_blocks = []
    
    logger.info(f"Running script for question {qno + 1} (max attempts: {max_tries})")
    logger.info(f"Question: {p.questions_list[qno]}")
    
    dfs_copy = [df.copy(deep=True) for df in p.dfs]
    
    for attempt in range(max_tries):
        try:
            env = {
                'pd': pd,
                'np': np,
                'dfs': dfs_copy,  # Use isolated copies
                '__builtins__': builtins
            }
            exec(function_def, env)

            if "find_answer" not in env:
                raise NameError("The function 'find_answer' is not defined in the script.")
                
            answer = env["find_answer"]() 
            logger.info(f"Successfully executed script on attempt {attempt + 1}")
            print(function_def)
            return answer
        
        except Exception as e:
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            full_traceback = ''.join(tb_lines)
            sanitized_traceback = re.sub(r'File ".*?",', 'File "<redacted>",', full_traceback)
            logger.info(f"Error in question script (attempt {attempt + 1})")
        
            if attempt < max_tries - 1:
                prompt_text = FIX_QUESTION_SCRIPT_TEMPLATE.format(
                    data_source_text = p.data_source_text,
                    questions_list_text = p.questions_list_text,
                    dfs_text = p.dfs_text,
                    script = function_def,
                    traceback = sanitized_traceback,
                    fix_history = '\n-'.join(fix_history_blocks) if fix_history_blocks else 'None',
                    question_string = p.questions_list[qno]
                )
                logger.info(f"Fix prompt for attempt {attempt + 1}:\n{prompt_text}")
                
                res = ask_LLM(
                    contents = [prompt_text],
                    response_schema = FIX_QUESTION_SCRIPT_SCHEMA,
                )
                function_def = res.get("fixed_function", "")
                fix_description = res.get("fix_description", "").strip()
                logger.info(f"Received fixed 'find_answer' function for attempt {attempt + 1}:\n{function_def}")
                logger.info(f"Fix description: {fix_description}")
                
                fix_history_blocks.append(f"Attempt {attempt + 1}: {fix_description}")
            else:
                logger.error(f"Max attempts reached for question {qno + 1}")
                return None 

def generate_output(p: Problem, answers):
    from prompts import OUTPUT_SCRIPT_TEMPLATE, OUTPUT_SCRIPT_SCHEMA
    
    logger.info("Generating final output")
    prompt_text = OUTPUT_SCRIPT_TEMPLATE.format(
        questions_list_text = p.questions_list_text,
        answers_list_text = create_answers_list_text(answers),
        output_format_text = p.output_format_text
    )
    logger.info(f"Output generation prompt:\n{prompt_text}")

    res = ask_LLM(contents=[prompt_text], response_schema=OUTPUT_SCRIPT_SCHEMA)
    function_def = res.get("function_definition")
    logger.info(f"Output generation function:\n{function_def}")
    try:
        env = {
            'pd': pd,
            'np': np,
            '__builtins__': builtins
        }
        exec(function_def, env)
        output = env["create_output"](answers) # type: ignore
        logger.info(f"Generated output: {str(output)[:200]}")
        return output
    except Exception as e:
        logger.info(f"Error generating output: {str(e)}")
        return {}  

def ask_LLM(contents: list, response_schema: dict) -> Any:
    models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite"
    ]
    client = genai.Client(api_key=os.environ.get("GEMINI_AUTH_TOKEN"))

    for model_name in models:
        try:
            logger.info(f"Trying model: {model_name}")
            config = {}
            if model_name in ["gemini-2.5-pro", "gemini-2.5-flash"]:
                config = {
                    "response_mime_type": "application/json",
                    "response_json_schema": response_schema,
                    "thinking_config": {
                        "thinking_budget": 0 if model_name == "gemini-2.5-flash" else 128  
                    }
                }
            else:
                config = {
                    "response_mime_type": "application/json",
                    "response_json_schema": response_schema
                }
            
            response = client.models.generate_content(
                model=model_name,
                contents=contents, 
                config=config # type: ignore
            )            
            if not response.text:
                raise ValueError("'response.txt' is None")
            
            response_json = json.loads(response.text)
            logger.info(f"Successfully got response from {model_name}")
            return response_json
        except Exception as e:
            logger.info(f"Model {model_name} failed: {str(e)[:100]}")
            continue
    
    logger.error("All gemini models failed")
    raise Exception("all gemini models failed")


def create_dfs_text(dfs) -> str:
    if not dfs:
        logger.info("No dataframes to create text from")
        return ''
    
    from prompts import DFS_TEXT_TEMPLATE
    logger.info(f"Creating text for {len(dfs)} dataframes")

    all_snippets_text = ""
    for i, df in enumerate(dfs):
        # truncate cell values 
        df = df.map(lambda x: (str(x)[:30] + '...') if isinstance(x, str) and len(x) > 30 else x)

        # create markdown snippet 
        snippet_text = df.sample(min(3, df.shape[0])).to_markdown()
        all_snippets_text += f"\ndfs[{i}] snippet:\n{snippet_text}\n"
        logger.info(f"Added snippet for dataframe {i}")

    dfs_text = DFS_TEXT_TEMPLATE.format(all_snippets_text=all_snippets_text)
    logger.info(f"Created dfs text with length {len(dfs_text)}")
    return dfs_text 

def create_answers_list_text(answers):
    text = ""
    for ans in answers:
        ans = str(ans)
        ans = f"{ans[:20]}...(truncated)" if len(ans) > 20 else ans 
        text += f"{ans},\n"
    return text


def json_compatible(obj):
    if obj is None:
        return None
    try:
        return obj.item()
    except:
        pass 

    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    if isinstance(obj, np.ndarray):
        return [json_compatible(x) for x in obj.tolist()]
    
    if isinstance(obj, (pd.Series, pd.Index)):
        return [json_compatible(x) for x in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        return obj.map(json_compatible).to_dict(orient='records')
    
    # Handle collections
    if isinstance(obj, dict):
        return {k: json_compatible(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [json_compatible(x) for x in obj]

    return str(obj)

def yaml_dump(data):
    return yaml.dump(data, sort_keys=False, default_flow_style=False, indent=4)
