import os
import json
import asyncio
import random
import time
from dotenv import load_dotenv
from google.genai import Client, types


EXHAUST_COOLDOWN = 30  # seconds
MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash"
]

# Load API keys
load_dotenv(".venv/secrets.env")
API_KEYS = [os.environ.get(f"API_KEY_{i}") for i in range(0, 6) if os.environ.get(f"API_KEY_{i}")]

# Global exhausted combo tracker
exhausted_combos = set() 
exhausted_lock = asyncio.Lock() 


async def mark_exhausted_temporarily(kidx, model):
    combo = (kidx, model)
    async with exhausted_lock: # ✨ Acquire lock to safely modify the set
        exhausted_combos.add(combo)
    print(f"[RESOURCE_EXHAUSTED] Marked {model} with API_KEY_{kidx} as exhausted for 30s")
    await asyncio.sleep(EXHAUST_COOLDOWN)
    async with exhausted_lock: # ✨ Acquire lock to safely modify the set again
        exhausted_combos.discard(combo)
    print(f"[COOLDOWN ENDED] {model} with API_KEY_{kidx} is now available again")


async def ask_gemini(contents: list, response_json_schema: dict):
    for model in MODELS:
        for kidx, api_key in enumerate(API_KEYS):
            key_model = (kidx, model)

            async with exhausted_lock: # 
                is_exhausted = key_model in exhausted_combos

            if is_exhausted:
                print(f"[SKIP] {model} with API_KEY_{kidx} is temporarily exhausted.")
                continue

            try:
                print(f"Trying {model} with API_KEY_{kidx}")
                client = Client(api_key=api_key)
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        contents=contents,
                        model=model,
                        config=_get_config(model, response_json_schema),
                    ),
                    timeout=40,
                )

                if not response.text:
                    raise ValueError("LLM response has no text")
                response_json = json.loads(response.text)
                print(f"[SUCCESS] {model} with API_KEY_{kidx}")
                return response_json

            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    asyncio.create_task(mark_exhausted_temporarily(kidx, model))

                print(f"[FAILED] {model} with API_KEY_{kidx}: {error_str[:25]+"..."}")
                continue

    raise Exception("Gemini failed to respond after trying all key-model combinations.")


def _get_config(model, response_json_schema):
    if model.startswith("gemini-2.5"):
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=response_json_schema,
            thinking_config=types.ThinkingConfig(thinking_budget=512),
        )
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=response_json_schema,
    )






