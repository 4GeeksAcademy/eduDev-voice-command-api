import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TaskCreate(ApiModel):
    title: NonBlankText
    done: bool = False


class TaskReplace(ApiModel):
    title: NonBlankText
    done: bool


class TaskUpdate(ApiModel):
    title: NonBlankText | None = None
    done: bool | None = None

    @model_validator(mode="after")
    def require_update(self) -> "TaskUpdate":
        if any(
            field in self.model_fields_set and getattr(self, field) is None
            for field in ("title", "done")
        ):
            raise ValueError("Provided task fields cannot be null.")
        if self.title is None and self.done is None:
            raise ValueError("At least one of 'title' or 'done' is required.")
        return self


class Task(ApiModel):
    id: int
    title: NonBlankText
    done: bool


class InstructionRequest(ApiModel):
    transcription: NonBlankText


class InstructionPayload(ApiModel):
    endpoint: str = Field(..., min_length=1)
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_allowed_operation(self) -> "InstructionPayload":
        item_match = re.fullmatch(r"/tasks/([1-9]\d*)", self.endpoint)
        if self.method in {"GET", "POST"}:
            if self.endpoint != "/tasks":
                raise ValueError(f"{self.method} is only allowed on /tasks.")
        elif item_match is None:
            raise ValueError(f"{self.method} requires /tasks/{{positive_integer}}.")

        payload_model: type[ApiModel] | None = {
            "POST": TaskCreate,
            "PUT": TaskReplace,
            "PATCH": TaskUpdate,
        }.get(self.method)
        if payload_model is None:
            if self.params:
                raise ValueError(f"{self.method} does not accept parameters.")
            self.params = {}
        else:
            validated_params = payload_model.model_validate(self.params)
            self.params = validated_params.model_dump(exclude_unset=True)
        return self


class TranscribeFlowResponse(ApiModel):
    transcription: str = Field(..., min_length=1)
    instruction: InstructionPayload
    result: Any
