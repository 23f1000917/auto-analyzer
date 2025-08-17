import os 
import re
import yaml
import json 
import glob
import logging
import traceback
import pandas as pd 
from PIL import Image
from io import BytesIO
from typing import Any
from datetime import datetime
from google import genai
from textwrap import dedent 
from dotenv import load_dotenv 
from PIL.ImageFile import ImageFile
from fastapi import FastAPI, Request
from starlette.datastructures import UploadFile

# ------------------------------------------ Setup Logging --------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------------------  FastAPI --------------------------------------------------

# load secrets
load_dotenv(".venv/secrets.env")

# initialize app
app = FastAPI()

@app.post("/analyze")
async def analyze(request: Request):
    logger.info("Starting analysis request")
    start_time = datetime.now()
    
    try:
        # create problem instance 
        problem = await create_problem_instance(request)  
        logger.debug(f"Problem instance created with properties: {vars(problem).keys()}")

        # use LLM to create metadata_dict then create metadata_text using yaml 
        logger.info("Generating problem metadata")
        problem.metadata_dict = generate_problem_metadata(problem) 
        problem.metadata_text = yaml_dump(problem.metadata_dict)
        logger.debug(f"Generated metadata:\n{problem.metadata_text}")

        # use metadata to populate other properties
        problem.questions_list = problem.metadata_dict.get("questions", [])
        problem.questions_list_text = yaml_dump(problem.questions_list)
        problem.data_source_text = problem.metadata_dict.get("data_source_text", "")
        problem.output_format_text = problem.metadata_dict.get("output_format_text", "")
        logger.debug(f"Questions list: {problem.questions_list}")
        logger.debug(f"Data source: {problem.data_source_text}")
        logger.debug(f"Output format: {problem.output_format_text}")

        # load dfs from attached files (if any)
        if len(problem.file_dict) > 0:
            logger.info(f"Loading {len(problem.file_dict)} attached files as dataframes")
            problem.files_dfs = load_files_as_dfs(problem)
            logger.debug(f"Loaded {len(problem.files_dfs)} dataframes from files")
        else:
            logger.info("No files attached for dataframe loading")

        # scrape tables from webpage if needed 
        logger.info("Checking if web scraping is needed")
        problem.scraped_dfs = webscrape_tables_if_needed(problem)
        if problem.scraped_dfs:
            logger.debug(f"Scraped {len(problem.scraped_dfs)} tables from web")
        else:
            logger.debug("No tables scraped from web")

        # combine dfs from both sources into dfs list and create dfs_text 
        problem.dfs = problem.files_dfs + problem.scraped_dfs
        problem.dfs_text = create_dfs_text(problem.dfs)
        logger.debug(f"Combined {len(problem.dfs)} dataframes available")
        logger.debug(f"Dataframes text preview:\n{problem.dfs_text[:200]}...")

        # find the answers to the questions
        logger.info(f"Finding answers to {len(problem.questions_list)} questions")
        answers = find_question_answers(problem)
        logger.debug(f"Generated answers: {answers}")

        output = generate_output(problem, answers)
        logger.info(f"Successfully generated output in {(datetime.now() - start_time).total_seconds():.2f} seconds")

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    finally:
        # Clean up request files
        files = glob.glob("request_data/*")
        for f in files:
            try:
                os.remove(f)
                logger.debug(f"Deleted temporary file: {f}")
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")
            
    return output

# ------------------------------------------ Main Pipeline --------------------------------------------------

class Problem:
    def __init__(self) -> None:
        logger.debug("Initializing new Problem instance")
        # The properties suffixed with 'text' are to be used with LLM prompts
        self.questions_text: str = ""                  # 'questions.txt' content      
        self.image_dict : dict[str, ImageFile] = {}    # attached images 
        self.file_dict: dict[str, BytesIO] = {}        # other attached files 

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
        logger.debug(f"Found {len(uploads)} upload files in request")

        problem = Problem() # create problem instance

        for name, upload_file in uploads:
            logger.debug(f"Processing upload file: {name} (type: {upload_file.content_type})")
            if name == "questions.txt":
                problem.questions_text = (await upload_file.read()).decode("utf-8")
                logger.debug(f"Loaded questions text: {problem.questions_text[:100]}...")

            elif str(upload_file.content_type).startswith("image/"):
                problem.image_dict[name] = Image.open(BytesIO(await upload_file.read()))
                logger.debug(f"Added image to image_dict: {name}")

            else:
                file_content = await upload_file.read()
                os.makedirs("request_data", exist_ok=True) 
                with open(f"request_data/{name}", "wb") as f:
                    f.write(file_content)
                problem.file_dict[name] = file_content
                logger.debug(f"Added file to file_dict: {name}")

        logger.info(f"Problem instance created with {len(problem.image_dict)} images and {len(problem.file_dict)} files")
        return problem

    except Exception as e:
        logger.error(f"Error creating problem instance: {str(e)}")
        raise

def generate_problem_metadata(p: Problem) -> Any:
    from prompts import PROBLEM_METADATA_TEMPLATE, PROBLEM_METADATA_SCHEMA

    logger.info("Generating problem metadata")
    attachments_text = "" # additional text for prompt that gives info about attached images/files  

    images = list(p.image_dict.values()) 
    if len(images) > 0:  
        attachments_text += f"You have been given {len(images)} images with the problem." 
        logger.debug(f"Included {len(images)} images in metadata prompt")

    filenames = list(p.file_dict.keys()) 
    if len(filenames) > 0:
        attachments_text += f"Names of files attached with this problem: \n-{'\n'.join(filenames)}"
        logger.debug(f"Included {len(filenames)} filenames in metadata prompt")

    prompt_text = PROBLEM_METADATA_TEMPLATE.format(questions_text = p.questions_text, attachments_text = attachments_text)
    logger.debug(f"Metadata prompt text:\n{prompt_text[:200]}...")

    response_json = ask_LLM(contents = [prompt_text] + images, response_schema = PROBLEM_METADATA_SCHEMA)
    logger.debug(f"Metadata response:\n{yaml_dump(response_json)}")
    
    return response_json

def load_files_as_dfs(p: Problem) -> list[pd.DataFrame]:
    from prompts import FILE_LOADING_PROMPT, FILE_LOADING_SCHEMA

    logger.info("Loading files as dataframes")
    prompt_text = FILE_LOADING_PROMPT.format(data_source_text = p.data_source_text)
    logger.debug(f"File loading prompt:\n{prompt_text[:200]}...")

    response_json = ask_LLM(contents=[prompt_text], response_schema=FILE_LOADING_SCHEMA) 
    script = response_json.get("script")
    logger.debug(f"Received script for file loading:\n{script[:200]}...")

    if not script: 
        logger.warning("No script returned for file loading")
        return []
    
    files_dfs = run_file_loading_script(script, p.file_dict)
    valid_dfs = [df for df in files_dfs if isinstance(df, pd.DataFrame) and not df.empty]
    logger.info(f"Successfully loaded {len(valid_dfs)} valid dataframes from files")
    
    return valid_dfs

def run_file_loading_script(script: str, files: dict, max_tries: int = 5) -> list:
    from prompts import FIX_FILE_LOADING_TEMPLATE, FIX_FILE_LOADING_SCHEMA
    fix_history_blocks = []

    logger.info(f"Running file loading script (max attempts: {max_tries})")
    
    for attempt in range(max_tries):
        try:
            logger.debug(f"Attempt {attempt + 1}/{max_tries} to run file loading script")
            locals_dict = {}
            exec(script, {}, locals_dict)
            files_dfs = locals_dict.get("files_dfs", [])
            
            if isinstance(files_dfs, list):
                logger.info(f"Successfully executed file loading script on attempt {attempt + 1}")
                return files_dfs
            else:
                logger.warning("files_dfs is not a list")
                return []

        except Exception as e:
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            full_traceback = ''.join(tb_lines)
            sanitized_traceback = re.sub(r'File ".*?",', 'File "<redacted>",', full_traceback)
            logger.warning(f"[Script attempt {attempt + 1}/{max_tries}] Error:\n{sanitized_traceback}")

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
            logger.debug(f"Fix prompt for attempt {attempt + 1}:\n{fix_prompt[:200]}...")

            res = ask_LLM(contents=[fix_prompt], response_schema=FIX_FILE_LOADING_SCHEMA)
            script = res.get("fixed_script", "")
            fix_description = res.get("fix_description", "").strip()
            logger.debug(f"Received fixed script for attempt {attempt + 1}:\n{script[:200]}...")
            logger.debug(f"Fix description: {fix_description}")

            fix_history_blocks.append(f"Attempt {attempt + 1}: {fix_description}")

    logger.error("All attempts to fix the script failed")
    return []

def webscrape_tables_if_needed(p: Problem) -> list[pd.DataFrame]:
    from prompts import WEBSCRAPE_URL_TEMPLATE, WEBSCRAPE_URL_SCHEMA

    logger.info("Checking if web scraping is needed")
    prompt_text = WEBSCRAPE_URL_TEMPLATE.format(data_source_text = p.data_source_text)
    logger.debug(f"Web scrape prompt:\n{prompt_text[:200]}...")

    response_json = ask_LLM(contents = [prompt_text], response_schema = WEBSCRAPE_URL_SCHEMA)
    URL = response_json.get("URL")
    logger.debug(f"Received URL for scraping: {URL}")

    if not URL: 
        logger.info("No URL provided for scraping")
        return []
    
    try: 
        logger.info(f"Attempting to scrape tables from URL: {URL}")
        scraped_tables = pd.read_html(URL)
        logger.debug(f"Found {len(scraped_tables)} potential tables")
        
        valid_tables = [table for table in scraped_tables if not table.empty and 
            any(not str(col).startswith("Unnamed") and not isinstance(col, int) for col in table.columns)]
        logger.info(f"Found {len(valid_tables)} valid tables after filtering")
        
        return valid_tables
    except ValueError as e:
        logger.warning(f"Failed to scrape tables from URL: {str(e)}")
        return []

def find_question_answers(p: Problem) -> list:
    from prompts import QUESTION_SCRIPTS_TEMPLATE, QUESTION_SCRIPTS_SCHEMA

    logger.info(f"Finding answers to {len(p.questions_list)} questions")
    prompt_text = QUESTION_SCRIPTS_TEMPLATE.format(metadata_text = p.metadata_text, dfs_text = p.dfs_text)
    logger.debug(f"Question answering prompt:\n{prompt_text[:200]}...")

    response_json = ask_LLM(contents=[prompt_text],response_schema=QUESTION_SCRIPTS_SCHEMA)
    scripts = response_json.get("scripts", [])
    logger.debug(f"Received {len(scripts)} answer scripts for questions")

    answers = []
    for qno, script in enumerate(scripts):
        logger.debug(f"Processing question {qno + 1}/{len(scripts)}")
        answer = run_question_script(script, p, qno, max_tries=5)
        answers.append(answer)
        logger.debug(f"Answer for question {qno + 1}: {str(answer)[:100]}...")

    return answers

def run_question_script(script: str, p: Problem, qno: int, max_tries: int = 10) -> Any:
    from prompts import FIX_QUESTION_SCRIPT_TEMPLATE, FIX_QUESTION_SCRIPT_SCHEMA
    fix_history_blocks = []
    
    logger.info(f"Running script for question {qno + 1} (max attempts: {max_tries})")
    logger.debug(f"Question: {p.questions_list[qno]}")
    
    for attempt in range(max_tries):
        try:
            logger.debug(f"Attempt {attempt + 1}/{max_tries} for question {qno + 1}")
            local_scope = {}
            exec(script, {"dfs": p.dfs}, local_scope)
            answer = local_scope.get("answer")
            logger.debug(f"Successfully executed script on attempt {attempt + 1}")
            return answer
        
        except Exception as e:
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            full_traceback = ''.join(tb_lines)
            sanitized_traceback = re.sub(r'File ".*?",', 'File "<redacted>",', full_traceback)
            logger.warning(f"Error in question script (attempt {attempt + 1}):\n{sanitized_traceback}")
        
            if attempt < max_tries:
                prompt_text = FIX_QUESTION_SCRIPT_TEMPLATE.format(
                    metadata_text = p.metadata_text,
                    dfs_text = p.dfs_text,
                    script = script,
                    traceback = sanitized_traceback,
                    fix_history = '\n-'.join(fix_history_blocks) if fix_history_blocks else 'None',
                    question_string = p.questions_list[qno]
                )
                logger.debug(f"Fix prompt for attempt {attempt + 1}:\n{prompt_text[:200]}...")
                
                res = ask_LLM(
                    contents = [prompt_text],
                    response_schema = FIX_QUESTION_SCRIPT_SCHEMA,
                )
                script = res.get("fixed_script", "")
                fix_description = res.get("fix_description", "").strip()
                logger.debug(f"Received fixed script for attempt {attempt + 1}:\n{script[:200]}...")
                logger.debug(f"Fix description: {fix_description}")
                
                fix_history_blocks.append(f"Attempt {attempt + 1}: {fix_description}")
            else:
                logger.error(f"Max attempts reached for question {qno + 1}")
                return None

def generate_output(p: Problem, answers):
    from prompts import OUTPUT_SCRIPT_TEMPLATE, OUTPUT_SCRIPT_SCHEMA

    answers_list_text = ", ".join([
        f"<{str(type(a)).split('.')[-1][:-2]}> {str(a)[:20]}{'...(truncated)' if len(str(a)) > 20 else ''}"
        for a in answers
    ])
    logger.info("Generating final output")
    prompt_text = OUTPUT_SCRIPT_TEMPLATE.format(
        questions_list_text = p.questions_list_text,
        answers_list_text = answers_list_text,
        output_format_text = p.output_format_text
    )
    logger.debug(f"Output generation prompt:\n{prompt_text[:200]}...")

    res = ask_LLM(contents=[prompt_text], response_schema=OUTPUT_SCRIPT_SCHEMA)
    script = res.get("function_definition")
    logger.debug(f"Output generation script:\n{script[:200]}...")

    try:
        local_ns = {}
        exec(script, {}, local_ns)
        create_output_func = local_ns.get("create_output", lambda answers: answers)
        output = create_output_func(answers)
        logger.debug(f"Generated output: {output}")
        return output
    except Exception as e:
        logger.error(f"Output generation failed: {str(e)}")
        return {"error": f"output generation failed", "answers": answers}


def ask_LLM(contents: list, response_schema: dict) -> Any:
    models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash"
    ]
    client = genai.Client(api_key=os.environ.get("GEMINI_AUTH_TOKEN"))
    
    logger.debug(f"Asking LLM with {len(contents)} content parts (first part: {contents[0][:200]}...)")
    logger.debug(f"Response schema: {response_schema}")
    
    for model_name in models:
        try:
            logger.debug(f"Trying model: {model_name}")
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
                config=config
            )
            
            if not response.text:
                raise ValueError("'response.txt' is None")
            
            response_json = json.loads(response.text)
            logger.debug(f"Successfully got response from {model_name}")
            logger.debug(f"Response preview:\n{yaml_dump(response_json)[:200]}...")
            
            return response_json
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {str(e)}")
            continue
    
    logger.error("All gemini models failed")
    raise Exception("all gemini models failed")


def create_dfs_text(dfs) -> str:
    if not dfs:
        logger.debug("No dataframes to create text from")
        return ''
    
    from prompts import DFS_TEXT_TEMPLATE
    logger.debug(f"Creating text for {len(dfs)} dataframes")

    all_snippets_text = ""
    for i, df in enumerate(dfs):
        # truncate cell values 
        df = df.map(lambda x: (str(x)[:30] + '...') if isinstance(x, str) and len(x) > 30 else x)

        # create markdown snippet 
        snippet_text = df.sample(min(3, df.shape[0])).to_markdown()
        all_snippets_text += f"\ndfs[{i}] snippet:\n{snippet_text}\n"
        logger.debug(f"Added snippet for dataframe {i}")

    dfs_text = DFS_TEXT_TEMPLATE.format(all_snippets_text=all_snippets_text)
    logger.debug(f"Created dfs text with length {len(dfs_text)}")
    return dfs_text 

def yaml_dump(data):
    return yaml.dump(data, sort_keys=False, default_flow_style=False, indent=4)
