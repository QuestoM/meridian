"""A measurement that ends while the shelf is being read still reads measuring.

**The defect this file exists to keep dead, and how it was found.** The shelf
follows a measurement by reading ``/api/model/candidates`` again while any card
reports one in flight, and it stops the moment none does. That makes the route's
answer the whole signal, so an answer that says nothing is in flight while
carrying the record from before the write ends the watch on a superseded figure.

Measured on the running instance on port 8037 on 2026-08-04, with a real
measurement started from outside the browser: the store recorded the finished
measurement at 15:22:24.511, the screen carried its money 0.21 s later with zero
presses behind it, and the record on the screen at that moment was the earlier
one from 15:09:32. The reads then stopped, correctly, because nothing was in
flight, and the screen kept the older record for as long as the reader stayed.
Both records were real and their figures agreed to the shekel, so nothing false
was on the screen that day. Nothing about the route guaranteed that.

The cause is an order. The measurement thread writes its record and then clears
its own entry in the register, and the route read the store first and the
register second, so a thread that finished between the two produced exactly that
answer. Reading the register first cannot: any record written before the
register was taken is already in the store read that follows it, and a thread
that finishes after it is still reported in flight, which costs one further read
a second and a half later.

This is a race in the ordinary sense and it cannot be reproduced by running the
real thing twice. So the interleaving is driven rather than waited for: the
store read itself performs the thread's own last two acts, in the thread's own
order, at the one instant that used to be fatal.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import model_console_api as api
from kairos_api import model_console_candidates as candidates
from kairos_api import model_version_store as store

ROOT = Path(__file__).resolve().parents[1]

# The shipped store, read for a real measurement record rather than a typed one.
SHIPPED_MEASUREMENTS = ROOT / "models" / "releases" / "candidate_measurements.json"


def _shipped_records() -> dict[str, Any]:
    if not SHIPPED_MEASUREMENTS.is_file():
        return {}
    records = json.loads(SHIPPED_MEASUREMENTS.read_text(encoding="utf-8"))
    return records if isinstance(records, dict) else {}


@pytest.fixture()
def subject() -> str:
    """A candidate that is on the shelf and carries a real stored measurement."""
    on_shelf = {candidates.candidate_id(path) for path in candidates.candidate_paths()}
    named = sorted(name for name in _shipped_records() if name in on_shelf)
    if not named:
        pytest.skip("no candidate on the shelf carries a stored measurement to read")
    return named[0]


@pytest.fixture()
def client(tmp_path, monkeypatch, subject) -> TestClient:
    """The console's routes, on a throwaway store seeded with that one record."""
    releases = tmp_path / "releases"
    releases.mkdir()
    seeded = {subject: _shipped_records()[subject]}
    (releases / store.MEASUREMENTS_FILE).write_text(
        json.dumps(seeded, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("KAIROS_MODEL_RELEASES_DIR", str(releases))
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


@pytest.fixture()
def register():
    """The in-flight register, emptied afterwards whatever the test did to it."""
    yield api._RUNNING  # noqa: SLF001 - this file is about that register
    with api._RUNNING_LOCK:  # noqa: SLF001
        api._RUNNING.clear()


def _finish_while_reading(monkeypatch, identifier: str) -> dict[str, int]:
    """Make the store read itself end the measurement, in the thread's own order.

    The thread saves its record and then clears the register. Both happen here,
    inside the store read the route performs, which is the one instant where the
    old order answered with the record from before the write and nothing in
    flight. The reads are counted, so a route that never reads the store fails
    rather than passing for the wrong reason.
    """
    calls = {"reads": 0}
    real = store.measurements

    def measurements_that_finish_first() -> dict[str, Any]:
        calls["reads"] += 1
        with api._RUNNING_LOCK:  # noqa: SLF001
            api._RUNNING.pop(identifier, None)
        return real()

    monkeypatch.setattr(store, "measurements", measurements_that_finish_first)
    return calls


def _start(register: dict[str, Any], identifier: str) -> None:
    register[identifier] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "past_durations_seconds": [],
    }


def _row(payload: dict[str, Any], identifier: str) -> dict[str, Any]:
    for row in payload.get("candidates") or []:
        if row.get("id") == identifier:
            return row
    raise AssertionError(f"the shelf carries no candidate called {identifier}")


def test_the_shelf_reports_measuring_when_the_measurement_ends_inside_its_own_read(
        client, register, subject, monkeypatch) -> None:
    """The watch survives the one interleaving that used to end it early."""
    calls = _finish_while_reading(monkeypatch, subject)
    _start(register, subject)
    payload = client.get("/api/model/candidates").json()
    assert calls["reads"] == 1, "the route answered without reading the store at all"
    assert _row(payload, subject)["money"]["state"] == "measuring", (
        "the shelf reported the measurement over while carrying the record from"
        " before it ended, which is the answer that stops the screen watching"
    )


def test_one_candidate_in_full_reports_the_same_thing_under_the_same_race(
        client, register, subject, monkeypatch) -> None:
    """The card's own route is read by the same screen and answers by the same rule."""
    calls = _finish_while_reading(monkeypatch, subject)
    _start(register, subject)
    payload = client.get(f"/api/model/candidates/{subject}").json()
    assert calls["reads"] >= 1, "the route answered without reading the store at all"
    assert payload["candidate"]["money"]["state"] == "measuring"


def test_the_measuring_answer_is_the_register_and_not_a_state_the_route_remembers(
        client, register, subject) -> None:
    """With nothing in flight the record's own state is what the shelf reports.

    Without this, a route that answered measuring for everything would pass the
    two above and leave the watch running for ever, which is the same defect
    wearing its fix.
    """
    settled = _row(client.get("/api/model/candidates").json(), subject)["money"]
    assert settled["state"] in {"measured", "stale"}, (
        f"a stored measurement reads {settled['state']} with nothing in flight"
    )
    _start(register, subject)
    assert _row(client.get("/api/model/candidates").json(), subject)["money"]["state"] == "measuring"
    with api._RUNNING_LOCK:  # noqa: SLF001
        api._RUNNING.clear()
    assert _row(client.get("/api/model/candidates").json(), subject)["money"] == settled, (
        "the shelf did not go back to the stored record once nothing was in flight"
    )
