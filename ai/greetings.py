"""
Greeting detection and responses.

Ported from the OXRO Streamlit project (Food_Delivery_Chatbot/app.py),
response-only. The Streamlit project itself is NOT modified.

Purpose:
    - Answer simple greetings locally so /chat does not need to spend
      Gemini quota on them.
    - Phase 5: handles English, Tamil script, Tanglish and mixed
      language greetings. Detection stays WHOLE-MESSAGE exact match,
      so a mixed message like "Vanakkam, enakku biryani venum"
      correctly routes to the food pipeline instead.
"""

from food_recommendation.language_phrases import normalize_message

# Tamil script greeting (vanakkam)
_TAMIL_VANAKKAM = "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd"


def detect_greeting(message):
    """
    True when the message is a standalone greeting such as
    "hi", "hello", "good morning", "vanakkam" or "வணக்கம்".
    """
    normalized = normalize_message(message)

    greetings = [
        "hi", "hello", "hey", "hii", "hiii",
        "good morning", "good afternoon",
        "good evening", "how are you",
        "hello there", "hi there",
        # Phase 5: Tanglish and Tamil script greetings
        "vanakkam", "vanakam",
        _TAMIL_VANAKKAM,
    ]

    return normalized in greetings


def greeting_response(message):
    """
    Friendly greeting reply matching the customer's greeting style.
    Tamil script greetings get a Tamil script reply, Tanglish
    (Latin "vanakkam") gets a Tanglish reply, English keeps the
    existing replies unchanged.
    """
    normalized = normalize_message(message)

    # Phase 5: Tamil script greeting -> Tamil script reply.
    if _TAMIL_VANAKKAM in normalized:
        return (
            "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd! "
            "OXRO-\u0b95\u0bcd\u0b95\u0bc1 "
            "\u0bb5\u0bb0\u0bb5\u0bc7\u0bb1\u0bcd\u0b95\u0bbf\u0bb1\u0bcb\u0bae\u0bcd. "
            "\u0b87\u0ba9\u0bcd\u0bb1\u0bc1 "
            "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bc1\u0b95\u0bcd\u0b95\u0bc1 "
            "\u0b8e\u0baa\u0bcd\u0baa\u0b9f\u0bbf "
            "\u0b89\u0ba4\u0bb5 "
            "\u0bae\u0bc1\u0b9f\u0bbf\u0baf\u0bc1\u0bae\u0bcd?"
        )

    # Phase 5: Tanglish greeting -> Tanglish reply.
    if "vanakkam" in normalized or "vanakam" in normalized:
        return (
            "Vanakkam! OXRO-kku varavetrikkirom. "
            "Inaiku enna sapadu venum?"
        )

    if "good morning" in normalized:
        return "Good morning! Welcome to OXRO. How can I help you today?"

    elif "good afternoon" in normalized:
        return "Good afternoon! Welcome to OXRO. How may I assist you?"

    elif "good evening" in normalized:
        return "Good evening! Welcome to OXRO. How can I help you?"

    elif "how are you" in normalized:
        return (
            "I'm doing great, thank you for asking! "
            "How can I assist you with your food delivery today?"
        )

    return "Hello! Welcome to OXRO Food Delivery Support. How may I help you today?"
