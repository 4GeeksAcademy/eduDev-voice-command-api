from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from src.app.schemas.voice import InstructionRequest, TranscribeFlowResponse
from src.app.services.executor import execute_instruction
from src.app.services.groq import (
    GroqProviderError,
    InvalidInstructionError,
    generate_instruction,
    transcribe_audio,
)
from src.app.utils.language import normalize_transcription_language

router = APIRouter(tags=["transcribe"])


@router.get("/")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/transcribe", response_model=TranscribeFlowResponse)
async def transcribe_and_run_flow(request: Request) -> TranscribeFlowResponse:
    transcription = await extract_transcription(request)
    try:
        instruction = await run_in_threadpool(generate_instruction, transcription)
        result = execute_instruction(instruction)
    except (GroqProviderError, InvalidInstructionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    return TranscribeFlowResponse(
        transcription=transcription,
        instruction=instruction,
        result=result,
    )


async def extract_transcription(request: Request) -> str:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("application/json"):
        try:
            payload = InstructionRequest.model_validate(await request.json())
        except (ValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="A non-empty 'transcription' string is required.",
            ) from exc
        return payload.transcription

    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Use multipart audio or JSON transcription.",
        )

    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Multipart field 'file' is required.",
        )
    contents = await upload.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded audio must not be empty.",
        )

    language = normalize_transcription_language(form.get("language"))
    try:
        return await run_in_threadpool(
            transcribe_audio,
            contents,
            upload.filename or "recording.webm",
            upload.content_type or "application/octet-stream",
            language,
        )
    except GroqProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
