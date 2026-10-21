
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.main import (
    BASE_DIR,
    FALLBACK_MODELS,
    MAX_MODEL_ATTEMPTS,
    MODEL_RETRY_DELAY,
    list_text_generation_models,
)


ENV_PATH = BASE_DIR / ".env"


def generate_answer_for_ui(prompt, client=None):
    """
    Generate an answer using an available Gemini text-generation model.

    Args:
        prompt: The prompt containing the question and document evidence.
        client: Optional Gemini client, useful for unit testing.

    Returns:
        A tuple containing (answer_text, model_name).

    Raises:
        RuntimeError: If the API key is missing or no model can generate
        a usable response.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The prompt must be a non-empty string.")

    if client is None:
        load_dotenv(ENV_PATH)

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY was not found in your .env file. "
                "Check the variable name without sharing the key."
            )

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=60000),
        )

    available_models = list_text_generation_models(client)

    models_to_try = []

    for model_name in available_models + FALLBACK_MODELS:
        if model_name not in models_to_try:
            models_to_try.append(model_name)

    models_to_try = models_to_try[:MAX_MODEL_ATTEMPTS]

    if not models_to_try:
        raise RuntimeError(
            "No Gemini text-generation models are configured."
        )

    last_error = None

    for index, model_name in enumerate(models_to_try):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1500,
                ),
            )

            answer = response.text

            if answer and answer.strip():
                return answer.strip(), model_name

            last_error = RuntimeError(
                f"{model_name} returned an empty answer."
            )

        except Exception as error:
            last_error = error

        if index < len(models_to_try) - 1:
            time.sleep(MODEL_RETRY_DELAY)

    raise RuntimeError(
        "Gemini could not generate an answer. Last error: "
        f"{type(last_error).__name__}: {last_error}"
    )
