import re
from typing import Any

from src.app.api.routes.tasks import (
    create_task,
    delete_task,
    get_tasks,
    replace_task,
    update_task,
)
from src.app.schemas.voice import (
    InstructionPayload,
    TaskCreate,
    TaskReplace,
    TaskUpdate,
)


def execute_instruction(instruction: InstructionPayload) -> Any:
    if instruction.method == "GET":
        return get_tasks()
    if instruction.method == "POST":
        return create_task(TaskCreate.model_validate(instruction.params))

    match = re.fullmatch(r"/tasks/(\d+)", instruction.endpoint)
    if match is None:
        raise ValueError("Instruction was not allowlisted.")
    task_id = int(match.group(1))

    if instruction.method == "PUT":
        return replace_task(task_id, TaskReplace.model_validate(instruction.params))
    if instruction.method == "PATCH":
        return update_task(task_id, TaskUpdate.model_validate(instruction.params))
    if instruction.method == "DELETE":
        return delete_task(task_id)
    raise ValueError("Instruction was not allowlisted.")
