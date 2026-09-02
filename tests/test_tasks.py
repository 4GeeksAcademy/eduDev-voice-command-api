from fastapi.testclient import TestClient


def test_create_and_list_tasks(client: TestClient) -> None:
    response = client.post("/tasks", json={"title": "  Buy milk  "})

    assert response.status_code == 201
    assert response.json() == {"id": 1, "title": "Buy milk", "done": False}
    assert client.get("/tasks").json() == [response.json()]


def test_ids_remain_monotonic_after_deletion(client: TestClient) -> None:
    client.post("/tasks", json={"title": "First"})
    client.post("/tasks", json={"title": "Second"})

    assert client.delete("/tasks/2").json() == {
        "message": "Task deleted successfully"
    }
    response = client.post("/tasks", json={"title": "Third"})

    assert response.json()["id"] == 3


def test_put_replaces_all_mutable_fields(client: TestClient) -> None:
    client.post("/tasks", json={"title": "Old title"})

    response = client.put(
        "/tasks/1", json={"title": "Replacement", "done": True}
    )

    assert response.status_code == 200
    assert response.json() == {"id": 1, "title": "Replacement", "done": True}


def test_patch_changes_only_provided_fields(client: TestClient) -> None:
    client.post("/tasks", json={"title": "Keep this", "done": False})

    response = client.patch("/tasks/1", json={"done": True})

    assert response.status_code == 200
    assert response.json() == {"id": 1, "title": "Keep this", "done": True}


def test_missing_task_returns_404_for_item_operations(client: TestClient) -> None:
    for method, payload in (
        ("put", {"title": "Missing", "done": False}),
        ("patch", {"done": True}),
        ("delete", None),
    ):
        response = client.request(method.upper(), "/tasks/99", json=payload)
        assert response.status_code == 404
        assert response.json() == {"detail": "Task not found"}


def test_task_body_validation(client: TestClient) -> None:
    assert client.post("/tasks", json={}).status_code == 422
    assert client.post("/tasks", json={"title": ""}).status_code == 422
    assert client.post("/tasks", json={"title": "   "}).status_code == 422
    assert client.post("/tasks", json={"title": 123}).status_code == 422
    assert (
        client.post("/tasks", json={"title": "Strict", "done": "yes"}).status_code
        == 422
    )
    assert client.put("/tasks/1", json={"title": "Incomplete"}).status_code == 422
    assert (
        client.put("/tasks/1", json={"title": "Strict", "done": 1}).status_code
        == 422
    )
    assert client.patch("/tasks/1", json={}).status_code == 422
    assert client.patch("/tasks/1", json={"done": None}).status_code == 422
    assert client.post("/tasks", json={"title": "Valid", "extra": 1}).status_code == 422
