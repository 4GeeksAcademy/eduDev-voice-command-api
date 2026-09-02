from fastapi import APIRouter, HTTPException, status

from src.app.schemas.voice import Task, TaskCreate, TaskReplace, TaskUpdate

router = APIRouter(prefix="/tasks", tags=["tasks"])

tasks: list[dict[str, object]] = []
next_task_id = 1


@router.get("", response_model=list[Task])
def get_tasks() -> list[Task]:
    return [Task.model_validate(task) for task in tasks]


@router.post("", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(payload: TaskCreate) -> Task:
    global next_task_id

    task = Task(id=next_task_id, **payload.model_dump())
    tasks.append(task.model_dump())
    next_task_id += 1
    return task


@router.put("/{task_id}", response_model=Task)
def replace_task(
    task_id: int,
    payload: TaskReplace,
) -> Task:
    task = find_task(task_id)
    task.update(payload.model_dump())
    return Task.model_validate(task)


@router.patch("/{task_id}", response_model=Task)
def update_task(
    task_id: int,
    payload: TaskUpdate,
) -> Task:
    task = find_task(task_id)
    task.update(payload.model_dump(exclude_none=True))
    return Task.model_validate(task)


@router.delete("/{task_id}")
def delete_task(task_id: int) -> dict[str, str]:
    task = find_task(task_id)
    tasks.remove(task)
    return {"message": "Task deleted successfully"}


def find_task(task_id: int) -> dict[str, object]:
    for task in tasks:
        if task["id"] == task_id:
            return task
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
