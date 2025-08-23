import os 
import shutil
import traceback
from fastapi import FastAPI, Request

from core import (
    create_problem_instance,
    generate_problem_metadata,
    load_files_as_dfs,
    webscrape_tables_if_needed,
    find_answers_to_questions,
    generate_output,
)


app = FastAPI()


@app.get("/")
def index():
    return {"message": "send a POST request with 'questions.txt'"}


@app.post("/analyze")
async def analyze(request: Request):
    try:
        p = await create_problem_instance(request)

        metadata = await generate_problem_metadata(p)
        p.questions = metadata["questions"]
        p.data_source_desc = metadata["data_source_desc"]
        p.output_format_desc = metadata["output_format_desc"]

        files_dfs = await load_files_as_dfs(p)
        scraped_dfs = await webscrape_tables_if_needed(p)

        p.dfs = files_dfs + scraped_dfs

        p.answers = await find_answers_to_questions(p)

        output = await generate_output(p)
        
        return output

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return {"error": "Analysis failed", "message": str(e)}
    finally:
        if p and os.path.exists(p.dir):
            shutil.rmtree(p.dir)

