import io
import os
import uuid
import builtins
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from fastapi import Request
from starlette.datastructures import UploadFile

import importlib
import subprocess
import traceback

import prompts
import schemas
import tracker
from gemini import ask_gemini


class Problem:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:6]
        self.dir = f"request_data/{self.id}"
        os.makedirs(self.dir, exist_ok=True)

        self.questions_text = ""
        self.images = []
        self.filenames = []

        self.questions: list[str] = []
        self.data_source_desc = ""
        self.output_format_desc = ""

        self.dfs = []

        self.answers = []


async def create_problem_instance(request: Request) -> Problem:
    form_items = (await request.form()).items()
    uploads = [(fn, val) for fn, val in form_items if isinstance(val, UploadFile)]
    p = Problem()

    for filename, upload_file in uploads:
        if filename == "questions.txt":
            p.questions_text = (await upload_file.read()).decode("utf-8")

        elif str(upload_file.content_type).startswith("image/"):
            p.images.append(Image.open(io.BytesIO(await upload_file.read())))
        else:
            file_content = await upload_file.read()

            with open(f"{p.dir}/{filename}", "wb") as f:
                f.write(file_content)

            p.filenames.append(filename)

    if p.questions_text == "":
        raise Exception("'questions.txt' is required")

    tracker.log_problem_instance(p)
    return p


async def generate_problem_metadata(p: Problem) -> dict:
    prompt_text = prompts.problem_metadata(
        questions_text=p.questions_text,
        images=p.images,
        filenames=p.filenames,
        problem_dir=p.dir,
    )
    response_json = await ask_gemini(
        contents=[prompt_text] + p.images, response_json_schema=schemas.problem_metadata
    )
    tracker.log_problem_metadata(response_json)
    return response_json


async def load_files_as_dfs(p: Problem) -> list[pd.DataFrame]:
    if len(p.filenames) == 0:
        return []

    prompt_text = prompts.files_dfs_script(
        data_source_desc=p.data_source_desc, problem_dir=p.dir
    )
    response_json = await ask_gemini(
        contents=[prompt_text], response_json_schema=schemas.files_dfs_script
    )
    _make_package_installations(response_json["packages"])
    script = response_json.get("script", None)
    if not script:
        raise ValueError("gemini response does not contain the script")
    files_dfs = await _run_files_dfs_script(script)
    valid_dfs = [
        df for df in files_dfs if isinstance(df, pd.DataFrame) and not df.empty
    ]
    tracker.log_dfs_from_files(p.filenames, valid_dfs)
    return valid_dfs


async def _run_files_dfs_script(script, max_tries=4) -> list[pd.DataFrame]:
    tried_fixes = []

    for attempt in range(max_tries):
        try:
            env = {"__builtins__": builtins}
            exec(script, env)
            files_dfs = env.get("files_dfs", None)

            if files_dfs is not None and isinstance(files_dfs, list):
                for i, df in enumerate(files_dfs):
                    if not isinstance(df, pd.DataFrame):
                        raise Exception(f"files_dfs[{i}] is not a dataframe")

                return files_dfs
            else:
                return []

        except Exception as e:
            if attempt < max_tries - 1:
                try_except_policy = "Do not use any try-except block in the script."
            else:
                try_except_policy = (
                    "Wrap each file load in try-except and skip failing ones."
                )
            prompt_text = prompts.fix_files_dfs_script(
                broken_script=script,
                tried_fixes=tried_fixes,
                try_except_policy=try_except_policy,
                e=e,
            )
            response_json = await ask_gemini(
                contents=[prompt_text],
                response_json_schema=schemas.fix_files_dfs_script,
            )
            script = response_json.get("fixed_script", None)
            if script is None:
                raise Exception("gemini response does not contain the fixed script")

            _make_package_installations(response_json["packages"])

            fix_description = response_json.get("fix_description", "").strip()
            tracker.log_script_attempt(
                purpose="files_dfs Creation",
                attempt_no=attempt,
                error_message=str(e),
                fix_description=fix_description,
            )
            tried_fixes.append(f"Attempt {attempt + 1}: {fix_description}")
    return []


async def webscrape_tables_if_needed(p: Problem) -> list[pd.DataFrame]:
    prompt_text = prompts.webscrape_url(data_source_desc=p.data_source_desc)

    response_json = await ask_gemini(
        contents=[prompt_text], response_json_schema=schemas.webscrape_url
    )
    URL = response_json.get("URL", None)
    try:
        scraped_tables = pd.read_html(URL)

        valid_tables = [
            table
            for table in scraped_tables
            if not table.empty
            and any(
                not str(col).startswith("Unnamed") and not isinstance(col, int)
                for col in table.columns
            )
        ]
        tracker.log_webscraped_tables(URL, valid_tables)
        return valid_tables

    except Exception:
        return []


async def find_answers_to_questions(p: Problem) -> list:
    prompt_text = prompts.find_answer_scripts(
        data_source_desc=p.data_source_desc,
        questions=p.questions,
        dfs=p.dfs,
    )
    response_json = await ask_gemini(
        contents=[prompt_text], response_json_schema=schemas.find_answer_scripts
    )
    scripts = response_json.get("scripts", None)

    if scripts is None or scripts == []:
        raise Exception("gemini response contained no scripts")
    _make_package_installations(response_json["packages"])
    answers = []

    for qno, script in enumerate(scripts):
        question_string = p.questions[qno]
        ans = await _run_find_answer_script(script, question_string, p)
        answers.append(_make_json_compliant(ans))
    return answers


async def _run_find_answer_script(
    script: str, question_string: str, p: Problem, max_tries: int = 4
):
    tried_fixes = []

    for attempt in range(max_tries):
        try:
            env = {"__builtins__": builtins}
            exec(script, env)
            find_answer_func = env.get("find_answer", None)

            if len(p.dfs) > 0:
                dfs_copy = [df.copy() for df in p.dfs]
                answer = find_answer_func(dfs_copy)  # type: ignore
            else:
                answer = find_answer_func()  # type: ignore
            tracker.log_final_find_answer_script(question_string, script)
            return answer

        except Exception as e:
            if attempt < max_tries - 1:
                prompt_text = prompts.fix_find_answer_script(
                    data_source_desc=p.data_source_desc,
                    target_question=question_string,
                    broken_script=script,
                    tried_fixes=tried_fixes,
                    dfs=p.dfs,
                    e=e,
                )
                response_json = await ask_gemini(
                    contents=[prompt_text],
                    response_json_schema=schemas.fix_find_answer_script,
                )
                script = response_json.get("fixed_script", "")
                fix_description = response_json.get("fix_description", "").strip()

                tried_fixes.append(f"Attempt {attempt + 1}: {fix_description}")
                tracker.log_script_attempt(
                    purpose=question_string,
                    attempt_no=attempt,
                    error_message=str(e),
                    fix_description=fix_description,
                )
                _make_package_installations(response_json["packages"])
            else:
                return None


async def generate_output(p: Problem, max_tries=4):
    tried_fixes = []

    prompt_text = prompts.output_script(
        questions=p.questions,
        answers=p.answers,
        output_format_desc=p.output_format_desc,
    )
    response_json = await ask_gemini(
        contents=[prompt_text], response_json_schema=schemas.output_script
    )
    script = response_json.get("script", None)
    if not script:
        raise Exception("gemini did not return a valid script")
    _make_package_installations(response_json["packages"])
    for attempt in range(max_tries):
        try:
            env = {"__builtins__": builtins}
            exec(script, env)
            output = env["create_output"](p.answers)  # type: ignore
            print(f"{'=' * 100}\n" f"Final Output Generation Script:\n" f"{script}")
            return output
        except Exception as e:
            if attempt < max_tries - 1:
                prompt_text = prompts.fix_output_script(
                    answers=p.answers,
                    questions=p.questions,
                    output_format_desc=p.output_format_desc,
                    broken_script=script,
                    e=e,
                )
                response = await ask_gemini(
                    contents=[prompt_text],
                    response_json_schema=schemas.fix_output_script,
                )
                script = response.get("fixed_script")
                if not script:
                    raise Exception("fixed script not found")
                fix_description = response.get("fix_description", "").strip()

                tried_fixes.append(f"Attempt {attempt + 1}: {fix_description}")
                tracker.log_script_attempt(
                    purpose="Output Creation",
                    attempt_no=attempt,
                    error_message=str(e),
                    fix_description=fix_description,
                )
            else:
                return str(e)


def _make_json_compliant(obj):
    if obj is None:
        return None
    try:
        return obj.item()
    except:
        pass

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, np.ndarray):
        return [_make_json_compliant(x) for x in obj.tolist()]

    if isinstance(obj, (pd.Series, pd.Index)):
        return [_make_json_compliant(x) for x in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        return obj.map(_make_json_compliant).to_dict(orient="records")

    if isinstance(obj, dict):
        return {k: _make_json_compliant(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_make_json_compliant(x) for x in obj]

    return str(obj)


def _make_package_installations(packages: list[str]) -> None:
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"⬇️ Installing package '{package}' using uv...")
            try:
                subprocess.run(
                    ["uv", "pip", "install", package],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                print(f"✅ Successfully installed '{package}'.")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install package '{package}'. Error:")
                print(e.stderr.strip())
            except Exception as e:
                print(f"❌ Unexpected error while installing '{package}': {e}")
                traceback.print_exception(type(e), e, e.__traceback__)
