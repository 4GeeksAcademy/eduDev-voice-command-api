from typing import Any

import groq
from groq import Groq
from pydantic import ValidationError

from src.app.core.config import get_settings
from src.app.schemas.voice import InstructionPayload


SYSTEM_PROMPT = """You route transcribed task commands to this API.
Return JSON only, with exactly this shape and no additional keys:
{"endpoint":"/tasks or /tasks/<integer>","method":"GET|POST|PUT|PATCH|DELETE","params":{}}

Allowed operations and parameter shapes:
- List tasks: GET /tasks with {}
- Create a task: POST /tasks with {"title":"string"}; "done" may be false
- Fully replace a task: PUT /tasks/<id> with both {"title":"string","done":boolean}
- Partially update a task: PATCH /tasks/<id> with at least one of {"title":"string","done":boolean}
- Delete a task: DELETE /tasks/<id> with {}

Examples:
"show my tasks" -> {"endpoint":"/tasks","method":"GET","params":{}}
"add buy milk" -> {"endpoint":"/tasks","method":"POST","params":{"title":"Buy milk"}}
"replace task 2 with call Ana and mark it done" -> {"endpoint":"/tasks/2","method":"PUT","params":{"title":"Call Ana","done":true}}
"mark task 2 done" -> {"endpoint":"/tasks/2","method":"PATCH","params":{"done":true}}
"rename task 2 to call Sam" -> {"endpoint":"/tasks/2","method":"PATCH","params":{"title":"Call Sam"}}
"delete task 2" -> {"endpoint":"/tasks/2","method":"DELETE","params":{}}

Never return prose, explanations, markdown, code fences, arbitrary URLs, or unsupported methods.
Choose the operation from the user's meaning. Do not use PUT unless all replacement fields are known.
"""


class GroqProviderError(Exception):
    pass


class InvalidInstructionError(Exception):
    pass


def get_groq_client() -> Groq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise GroqProviderError("Groq is not configured.")
    return Groq(
        api_key=settings.groq_api_key,
        timeout=settings.request_timeout_seconds,
    )


def generate_instruction(transcription: str) -> InstructionPayload:
    settings = get_settings()
    try:
        completion = get_groq_client().chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcription},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except groq.APIError as exc:
        raise GroqProviderError("Groq could not route the instruction.") from exc

    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise InvalidInstructionError("Groq returned an invalid instruction.") from exc
    if not content:
        raise InvalidInstructionError("Groq returned an empty instruction.")
    try:
        return InstructionPayload.model_validate_json(content)
    except ValidationError as exc:
        raise InvalidInstructionError("Groq returned an invalid instruction.") from exc


def transcribe_audio(
    contents: bytes,
    filename: str,
    content_type: str,
    language: str | None,
) -> str:
    settings = get_settings()
    request: dict[str, Any] = {
        "file": (filename, contents, content_type),
        "model": settings.groq_transcription_model,
    }
    if language is not None:
        request["language"] = language

    try:
        transcription = get_groq_client().audio.transcriptions.create(**request)
    except groq.APIError as exc:
        raise GroqProviderError("Groq could not transcribe the audio.") from exc

    try:
        text = transcription.text.strip()
    except (AttributeError, TypeError) as exc:
        raise GroqProviderError("Groq returned an invalid transcription.") from exc
    if not text:
        raise GroqProviderError("Groq returned an empty transcription.")
    return text
