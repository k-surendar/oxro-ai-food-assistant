"""
AI conversation module for the OXRO Food Assistant.

Migrated from the OXRO notebook (ai/conversation_manager.py).

Purpose:
    - Creates a Gemini chat session per customer with the OXRO
      system instruction.
    - Sends messages while maintaining conversation context.

Migration fix:
    - send_message() previously swallowed ALL exceptions and returned a
      generic string, which made quota (429) and server errors invisible
      to the API layer. It now raises typed exceptions so the FastAPI
      layer (Phase 3) can map them to proper HTTP responses.
"""

from google.genai import errors


class GeminiQuotaError(Exception):
    """Raised when the Gemini API quota is exhausted (HTTP 429)."""


class GeminiTransientError(Exception):
    """Raised when Gemini is temporarily unavailable (5xx errors)."""


class GeminiTimeoutError(Exception):
    """Raised when a Gemini call times out (HTTP 408 / client timeout)."""


class GeminiServiceError(Exception):
    """Raised for unexpected Gemini client failures."""


def create_food_chat(client):
    """
    Creates a conversational session for OXRO AI Food Assistant.

    Each returned chat object keeps its own message history, which is what
    gives per-customer conversation isolation.
    """
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config={
            "system_instruction": """
            You are OXRO AI Food Assistant.

            Help customers discover food and cuisines.
            Remember the conversation context.
            Understand follow-up questions.
            Respect vegetarian and non-vegetarian preferences.
            Never invent restaurant names, prices,
            availability, or order status.
            Reply in the same language the customer used
            (English, Tamil, Tanglish, or a mix).

            Be friendly, concise, and helpful.
            """
        }
    )

    return chat


def send_message(chat, user_message):
    """
    Sends a message while maintaining conversation context.

    Raises:
        GeminiQuotaError:     quota exhausted (429) - retry later.
        GeminiTransientError: Gemini temporarily unavailable (5xx).
        GeminiTimeoutError:   the call timed out (408 / client timeout).
        GeminiServiceError:   any other Gemini failure.
    """
    try:
        response = chat.send_message(user_message)
        return response.text

    except (
        GeminiQuotaError,
        GeminiTransientError,
        GeminiTimeoutError,
        GeminiServiceError,
    ):
        # Typed errors raised upstream pass through untouched so the API
        # layer maps them to the correct HTTP status.
        raise

    except errors.ClientError as exc:
        # ClientError carries the HTTP status of the Gemini call.
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
        raise GeminiTransientError(
            "Gemini is temporarily unavailable. Please try again shortly."
        ) from exc

    except TimeoutError as exc:
        # Built-in timeout (asyncio/HTTP client timeouts subclass this).
        raise GeminiTimeoutError(
            "Gemini request timed out. Please try again shortly."
        ) from exc

    except Exception as exc:
        raise GeminiServiceError(
            "Unexpected Gemini failure."
        ) from exc
