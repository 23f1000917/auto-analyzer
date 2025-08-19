import os
import json
import asyncio
from dotenv import load_dotenv
from google.genai import Client, types


MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

# load secrets
load_dotenv(".venv/secrets.env")

API_KEYS = [os.environ.get(f"API_KEY_{i}") for i in range(0, 6)]


async def ask_gemini(contents: list, response_json_schema: dict):
    print("=" * 100)
    print(contents[0])
    for model in MODELS:
        for kidx, api_key in enumerate(API_KEYS):
            try:
                print(f"trying {model} with api_key_{kidx}")
                client = Client(api_key=api_key)
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        contents=contents,
                        model=model,
                        config=_get_config(model, response_json_schema),
                    ),
                    timeout=30,
                )
                if not response.text:
                    raise ValueError("LLM response has no text")

                print(f"[SUCCESS] {model} with api_key_{kidx}")
                response_json = json.loads(response.text)
                return response_json

            except Exception as e:
                print(f"[FAILED] {model} with api_key_{kidx}: {str(e)}")
                print("FAILED RESPONSE TEXT: ")
                print(response.text)
                continue
    raise Exception("gemini failed to respond")


def _get_config(model, response_json_schema):
    config = None

    if model == "gemini-2.5-pro":
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=response_json_schema,
            thinking_config=types.ThinkingConfig(thinking_budget=128),
        )

    elif model == "gemini-2.5-flash":
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=response_json_schema,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
    else:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=response_json_schema,
        )
    return config
