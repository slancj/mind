import json
import os
from itertools import cycle

from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

VERY_WEAK_MODELS = [

    "gemini/gemma-4-26b-a4b-it",
]
_very_weak_cycle = cycle(VERY_WEAK_MODELS)

WEAK = ChatLiteLLM(model="gemini/gemma-4-31b-it", api_key=GEMINI_API_KEY)
MEDIUM = ChatLiteLLM(model="gemini/gemini-3.5-flash-lite", api_key=GEMINI_API_KEY)
STRONG = ChatLiteLLM(model="gemini/gemini-3.7-flash", api_key=GEMINI_API_KEY)


def complete(model: ChatLiteLLM, messages: list) -> str:
    response = model.invoke(messages)
    return extract_text(response)


def extract_text(response) -> str:
    content = response.content or ""
    if isinstance(content, str):
        return content
    parts = [b.get("text", "") for b in content if isinstance(b, dict)]
    return "".join(parts).strip()


def complete_very_weak(messages: list) -> str:
    model = ChatLiteLLM(model=next(_very_weak_cycle), api_key=GEMINI_API_KEY)
    return complete(model, messages)


def complete_weak(messages: list) -> str:
    return complete(WEAK, messages)


def complete_medium(messages: list) -> str:
    return complete(MEDIUM, messages)


def complete_strong(messages: list) -> str:
    return complete(STRONG, messages)


def complete_structured(model: ChatLiteLLM, messages: list, schema: dict) -> dict:
    response = model.invoke(
        messages,
        response_mime_type="application/json",
        response_schema=schema,
    )
    return json.loads(extract_text(response))


def complete_weak_structured(messages: list, schema: dict) -> dict:
    return complete_structured(WEAK, messages, schema)