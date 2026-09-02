from fastapi import APIRouter, HTTPException, status

from src.app.schemas.voice import InstructionPayload, InstructionRequest
from src.app.services.groq import (
    GroqProviderError,
    InvalidInstructionError,
    generate_instruction,
)

router = APIRouter(tags=["instruction"])


@router.post("/instruction", response_model=InstructionPayload)
def route_instruction(
    payload: InstructionRequest,
) -> InstructionPayload:
    try:
        return generate_instruction(payload.transcription)
    except GroqProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except InvalidInstructionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
