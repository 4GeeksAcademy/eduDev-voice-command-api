from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.app.api.routes import instruction as instruction_route
from src.app.api.routes import transcribe as transcribe_route
from src.app.schemas.voice import InstructionPayload
from src.app.services import groq as groq_service


def instruction(
    method: str = "GET",
    endpoint: str = "/tasks",
    params: dict[str, object] | None = None,
) -> InstructionPayload:
    return InstructionPayload(
        endpoint=endpoint,
        method=method,
        params=params or {},
    )


def test_instruction_returns_exact_routing_payload_without_mutation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = instruction("POST", "/tasks", {"title": "Buy milk"})
    monkeypatch.setattr(instruction_route, "generate_instruction", lambda _: expected)

    response = client.post(
        "/instruction", json={"transcription": "add buy milk"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "endpoint": "/tasks",
        "method": "POST",
        "params": {"title": "Buy milk"},
    }
    assert client.get("/tasks").json() == []


def test_instruction_rejects_whitespace_only_transcription(client: TestClient) -> None:
    response = client.post("/instruction", json={"transcription": "   "})

    assert response.status_code == 422


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        '{"endpoint":"https://example.com","method":"GET","params":{}}',
        '{"endpoint":"/tasks","method":"TRACE","params":{}}',
        '{"endpoint":"/tasks/1","method":"PUT","params":{"title":"Only"}}',
        '{"endpoint":"/tasks/1","method":"PATCH","params":{"done":"yes"}}',
        '{"endpoint":"/tasks","method":"POST","params":{"title":123}}',
    ],
)
def test_groq_output_must_be_valid_and_allowlisted(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: completion)
        )
    )
    monkeypatch.setattr(groq_service, "get_groq_client", lambda: client)

    with pytest.raises(groq_service.InvalidInstructionError):
        groq_service.generate_instruction("test command")


def test_valid_instruction_params_are_normalized() -> None:
    payload = InstructionPayload.model_validate(
        {
            "endpoint": "/tasks/1",
            "method": "PUT",
            "params": {"title": "  Buy milk  ", "done": False},
        }
    )

    assert payload.params == {"title": "Buy milk", "done": False}


def test_groq_instruction_call_uses_strict_json_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"endpoint":"/tasks","method":"GET","params":{}}'
                )
            )
        ]
    )

    def create(**request: object) -> SimpleNamespace:
        captured.update(request)
        return completion

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(groq_service, "get_groq_client", lambda: fake_client)
    monkeypatch.setattr(
        groq_service,
        "get_settings",
        lambda: SimpleNamespace(groq_model="configured-model"),
    )

    result = groq_service.generate_instruction("show my tasks")

    assert result == instruction("GET")
    assert captured["model"] == "configured-model"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {"role": "system", "content": groq_service.SYSTEM_PROMPT}
    assert "Return JSON only" in groq_service.SYSTEM_PROMPT
    assert "Never return prose" in groq_service.SYSTEM_PROMPT
    for operation in ("GET", "POST", "PUT", "PATCH", "DELETE"):
        assert operation in groq_service.SYSTEM_PROMPT
    assert messages[1] == {"role": "user", "content": "show my tasks"}


def test_json_transcription_fallback_executes_create(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        transcribe_route,
        "generate_instruction",
        lambda _: instruction("POST", "/tasks", {"title": "Buy milk"}),
    )

    response = client.post(
        "/transcribe", json={"transcription": "add buy milk"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "transcription": "add buy milk",
        "instruction": {
            "endpoint": "/tasks",
            "method": "POST",
            "params": {"title": "Buy milk"},
        },
        "result": {"id": 1, "title": "Buy milk", "done": False},
    }


def test_transcribe_flow_executes_list_update_and_delete(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post("/tasks", json={"title": "Initial"})
    routed = iter(
        [
            instruction("GET"),
            instruction("PATCH", "/tasks/1", {"done": True}),
            instruction("PUT", "/tasks/1", {"title": "Replaced", "done": False}),
            instruction("DELETE", "/tasks/1"),
        ]
    )
    monkeypatch.setattr(transcribe_route, "generate_instruction", lambda _: next(routed))

    listed = client.post("/transcribe", json={"transcription": "list"})
    patched = client.post("/transcribe", json={"transcription": "finish one"})
    replaced = client.post("/transcribe", json={"transcription": "replace one"})
    deleted = client.post("/transcribe", json={"transcription": "delete one"})

    assert listed.json()["result"] == [
        {"id": 1, "title": "Initial", "done": False}
    ]
    assert patched.json()["result"] == {
        "id": 1,
        "title": "Initial",
        "done": True,
    }
    assert replaced.json()["result"] == {
        "id": 1,
        "title": "Replaced",
        "done": False,
    }
    assert deleted.json()["result"] == {"message": "Task deleted successfully"}


def test_multipart_audio_uses_frontend_fields_and_optional_language(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_transcribe(
        contents: bytes, filename: str, content_type: str, language: str | None
    ) -> str:
        captured.update(
            contents=contents,
            filename=filename,
            content_type=content_type,
            language=language,
        )
        return "show my tasks"

    monkeypatch.setattr(transcribe_route, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(
        transcribe_route, "generate_instruction", lambda _: instruction("GET")
    )

    response = client.post(
        "/transcribe",
        files={"file": ("recording.webm", b"audio-bytes", "audio/webm")},
        data={"language": "EN-us"},
    )

    assert response.status_code == 200
    assert response.json()["transcription"] == "show my tasks"
    assert response.json()["result"] == []
    assert captured == {
        "contents": b"audio-bytes",
        "filename": "recording.webm",
        "content_type": "audio/webm",
        "language": "en",
    }


def test_groq_audio_transcription_uses_configured_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def create(**request: object) -> SimpleNamespace:
        captured.update(request)
        return SimpleNamespace(text="  add buy milk  ")

    fake_client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(groq_service, "get_groq_client", lambda: fake_client)

    result = groq_service.transcribe_audio(
        b"audio", "recording.webm", "audio/webm", "en"
    )

    assert result == "add buy milk"
    assert captured == {
        "file": ("recording.webm", b"audio", "audio/webm"),
        "model": "whisper-large-v3-turbo",
        "language": "en",
    }


def test_transcribe_request_validation(client: TestClient) -> None:
    assert client.post("/transcribe", json={}).status_code == 422
    assert client.post("/transcribe", json={"transcription": ""}).status_code == 422
    assert client.post("/transcribe", json={"transcription": "   "}).status_code == 422
    assert client.post("/transcribe", json={"transcription": 123}).status_code == 422
    assert (
        client.post(
            "/transcribe",
            content=b"plain text",
            headers={"content-type": "text/plain"},
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/transcribe", files={"other": ("other.txt", b"x", "text/plain")}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/transcribe",
            files={"file": ("empty.webm", b"", "audio/webm")},
        ).status_code
        == 422
    )


def test_provider_errors_are_controlled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_: str) -> InstructionPayload:
        raise groq_service.GroqProviderError("Groq could not route the instruction.")

    monkeypatch.setattr(instruction_route, "generate_instruction", fail)
    monkeypatch.setattr(transcribe_route, "generate_instruction", fail)

    instruction_response = client.post(
        "/instruction", json={"transcription": "list tasks"}
    )
    transcribe_response = client.post(
        "/transcribe", json={"transcription": "list tasks"}
    )

    assert instruction_response.status_code == 502
    assert instruction_response.json() == {
        "detail": "Groq could not route the instruction."
    }
    assert transcribe_response.status_code == 502
    assert transcribe_response.json() == {
        "detail": "Groq could not route the instruction."
    }
