"""P9: literal first answer text is local and precedes every external call."""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.assistant as assistant
from kairos_api import assistant_stream


def test_the_first_answer_text_precedes_the_worker(monkeypatch) -> None:
    started = []

    def body(*args, **kwargs):
        started.append(True)
        kwargs["on_stage"]("reading", {})
        kwargs["on_stage"]("grounded", {"facts": {"channel": "רשת 13"}})
        return {
            "available": True, "answer": "תשובה", "model": "test",
            "grounding": {}, "context_disclosure": {}, "truncated": False,
            "error": None, "proposals": None, "tool_trace": [],
            "conversation_id": "c-test",
        }

    monkeypatch.setattr(assistant, "_ask_body", body)
    monkeypatch.setattr(assistant, "_rate_limited", lambda: False)
    monkeypatch.setattr(assistant, "_audit_ask", lambda *args: None)
    monkeypatch.setattr(assistant, "_deadline_seconds", lambda: 45.0)
    monkeypatch.setattr(assistant_stream.assistant_memory, "append_entry", lambda *args, **kwargs: None)
    app = FastAPI()
    app.include_router(assistant.router)

    frames = TestClient(app).post(
        "/api/assistant/ask/stream", json={"question": "מה מצב השבוע"}
    ).text.split("\n\n")
    assert started == [True]
    assert frames[0].startswith("event: stage")
    grounded = next(index for index, frame in enumerate(frames) if '"stage": "grounded"' in frame)
    assert frames[grounded + 1].startswith("event: delta")
    payload = json.loads(frames[grounded + 1].split("data: ", 1)[1])
    assert payload == {"text": "בודק את השאלה מול נתוני המערכת. "}
    assert all(not frame.startswith("event: delta") for frame in frames[:grounded])


def test_the_opening_claims_no_result_and_matches_the_question_language() -> None:
    assert assistant_stream._opening_line("why is pacing late").startswith("Checking")
    hebrew = assistant_stream._opening_line("למה הקמפיין מאחר")
    assert hebrew == "בודק את השאלה מול נתוני המערכת. "
    assert not any(character.isdigit() for character in hebrew)
