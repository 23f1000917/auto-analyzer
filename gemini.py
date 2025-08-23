import os
import json
import asyncio
import random
import time
from dotenv import load_dotenv
from google.genai import Client, types


EXHAUST_COOLDOWN = 60  # seconds
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
exhausted_combos = set()  # Set of (api_key, model) tuples


async def mark_exhausted_temporarily(api_key, model):
    combo = (api_key, model)
    exhausted_combos.add(combo)
    print(f"[RESOURCE_EXHAUSTED] Marked {model} with key as exhausted for 60s")
    await asyncio.sleep(EXHAUST_COOLDOWN)
    exhausted_combos.discard(combo)
    print(f"[COOLDOWN ENDED] {model} with key is now available again")


async def ask_gemini(contents: list, response_json_schema: dict):
    print("=" * 100)
    print(contents[0])

    for model in MODELS:
        for kidx, api_key in enumerate(API_KEYS):
            key_model = (api_key, model)

            if key_model in exhausted_combos:
                print(f"[SKIP] {model} with api_key_{kidx} is temporarily exhausted.")
                continue

            try:
                print(f"Trying {model} with api_key_{kidx}")
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
                return json.loads(response.text)

            except Exception as e:
                error_str = str(e)
                if "RESOURCE_EXHAUSTED" in error_str:
                    asyncio.create_task(mark_exhausted_temporarily(api_key, model))
                    continue

                print(f"[FAILED] {model} with api_key_{kidx}: {error_str}")
                continue

    raise Exception("Gemini failed to respond after trying all key-model combinations.")


def _get_config(model, response_json_schema):
    if model.startswith("gemini-2.5"):
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=response_json_schema,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        )
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=response_json_schema,
    )
