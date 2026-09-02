import pytest
from fastapi.testclient import TestClient

from src.app.api.routes import tasks as tasks_route
from src.app.core.config import get_settings
from src.app.main import app


@pytest.fixture(autouse=True)
def reset_application_state() -> None:
    tasks_route.tasks.clear()
    tasks_route.next_task_id = 1
    get_settings.cache_clear()
    yield
    tasks_route.tasks.clear()
    tasks_route.next_task_id = 1
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
