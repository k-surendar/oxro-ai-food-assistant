"""
Catalogue-grounded AI recommendation module.

Merged from the OXRO notebook (ai/food_assistant.py + ai/recommendation_assistant.py).
The two notebook modules used near-identical catalogue prompts; this module
keeps the single, most hardened prompt (the full guardrail instruction list)
so there is one source of truth.

Purpose:
    - Run smart_food_search against the OXRO catalogue.
    - Pass ONLY the matched catalogue items to Gemini.
    - Return a structured result dict with clear statuses.

Fixes applied during migration:
    - No module-level `client` dependency: the Gemini client is passed in
      as a parameter (no monkey-patching, no globals).
    - Quota (429), server (5xx, retried with exponential backoff),
      timeout (408 / client timeout) and unexpected failures all raise
      the SAME typed errors as conversation_manager.send_message, so
      the API layer maps them identically for /chat and /recommend-food
      (429/503/504/502). Internal exception details are never placed
      into client-facing values.
    - The unexplained notebook `Status: error` was caused by stale module
      state after importlib.reload in Colab; merged module avoids that
      reload pattern entirely.
"""

import time

from google.genai import errors

from ai.conversation_manager import (
    GeminiQuotaError,
    GeminiServiceError,
    GeminiTimeoutError,
    GeminiTransientError,
)
from food_recommendation.food_search import smart_food_search


def recommend_food_with_ai(user_message, client, max_retries=3):
    """
    Generate a catalogue-grounded food recommendation.

    Returns a dict:
        {
            "status": "success" | "no_results",
            "preferences": {...},          # on success
            "food_results": [...],         # on success
            "ai_response": "..."           # always
        }

    Raises (mapped to HTTP statuses by the API layer):
        GeminiQuotaError:     quota exhausted (429) - not retried.
        GeminiTransientError: Gemini unavailable (5xx) after retries.
        GeminiTimeoutError:   the call timed out (408 / client timeout).
        GeminiServiceError:   any other Gemini failure.
    """
    search_result = smart_food_search(user_message)

    food_items = [
        item["food"]
        for item in search_result["results"]
    ]

    if not food_items:
        return {
            "status": "no_results",
            "ai_response": (
                "I couldn't find matching food items. "
                "Could you try another food request?"
            )
        }

    prompt = f"""
    You are OXRO AI Food Assistant.

    Customer request:
    {user_message}

    Detected preferences:
    {search_result["preferences"]}

    Available matching food items from OXRO database:
    {food_items}

    Instructions:
    - Recommend only food items from the available list.
    - Prioritize the most relevant items.
    - Respect vegetarian and non-vegetarian preferences.
    - Explain recommendations in friendly, simple language.
    - Reply in the same language the customer used (English, Tamil, Tanglish or a mix).
    - Do not invent restaurant names, prices or availability.
    - Do not promise food customization or preparation options.
    - Present customer preferences as preferences, not confirmed restaurant capabilities.
    - Recommend only information supported by the OXRO database.
    - Never claim that a food item is currently available, in stock, or orderable.
    - Say "listed in the OXRO food catalog" instead of "available".
    - Live availability must be confirmed by the backend.
    - Do not invent ingredients, nutritional values, or health benefits.
    - Do not describe a dish's ingredients unless they are provided by the database.
    - Do not claim a dish is healthy based only on its name.
    - If nutritional information is unavailable, clearly say so.
    - Recommend food based on the customer's preferences and available food names.
    """

    for attempt in range(max_retries):

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            return {
                "status": "success",
                "preferences": search_result["preferences"],
                "food_results": food_items,
                "ai_response": response.text
            }

        except (
            GeminiQuotaError,
            GeminiTransientError,
            GeminiTimeoutError,
            GeminiServiceError,
        ):
            # Typed errors raised upstream pass through untouched so the
            # API layer maps them to the correct HTTP status.
            raise

        except errors.ClientError as exc:
            status = getattr(exc, "code", None)

            if status == 429:
                raise GeminiQuotaError(
                    "Gemini quota exhausted. Please try again later."
                ) from exc

            if status == 408:
                raise GeminiTimeoutError(
                    "Gemini request timed out. Please try again shortly."
                ) from exc

            raise GeminiServiceError(
                f"Gemini rejected the request (status {status})."
            ) from exc

        except errors.ServerError as exc:

            if attempt == max_retries - 1:
                raise GeminiTransientError(
                    "Gemini is temporarily unavailable. "
                    "Please try again shortly."
                ) from exc

            time.sleep(2 ** attempt)

        except TimeoutError as exc:
            # Built-in timeout (asyncio/HTTP client timeouts subclass
            # this). Not retried, consistent with /chat.
            raise GeminiTimeoutError(
                "Gemini request timed out. Please try again shortly."
            ) from exc

        except Exception as exc:
            raise GeminiServiceError(
                "Unexpected Gemini failure."
            ) from exc
