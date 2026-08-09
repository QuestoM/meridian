"""The two reads that closed named coverage gaps: pricing a restriction, and access.

``estimate_restriction_cost`` answers the question a person asks before writing
a rule about somebody else's revenue: what does it cost. Kai could list the
constraints that exist and propose a new one, and could not price either.

``get_accounts`` is the account-administrator persona, which had zero coverage.
It also closes a measured asymmetry: the propose path already refuses a settings
change touching one of the four broadcast-licence limits, and no tool listed
them, so Kai could decline to move a number it could not name.

Both are reads and neither proposes anything. The tests below hold each to the
property that would fail silently: the restriction preview must write nothing
and must keep its two money bases apart, and the accounts tool must never carry
password material and must state its refusal rather than return an empty roster.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kairos_api import assistant_tools as tools
from kairos_api.assistant_read_tools import SOURCE_BY_TOOL, _READ_EXECUTORS, execute_read_tool

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / "data" / "placement_constraints.csv"


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _a_real_programme() -> str:
    from kairos_api.constraints_restrictions import restriction_titles

    rows = restriction_titles("").get("titles") or []
    live = [row for row in rows if int(row.get("planned_breaks") or 0) > 0]
    if not live:
        pytest.skip("no programme on the operator's channel carries a planned break")
    return str(max(live, key=lambda row: int(row["planned_breaks"]))["title"])


def _where(title: str) -> dict:
    return {"combinator": "and",
            "conditions": [{"field": "programme", "operator": "is", "value": title}]}


# --- both tools are registered like every other read ------------------------------
def test_both_reads_are_registered_with_an_executor_and_a_named_source() -> None:
    for name in ("estimate_restriction_cost", "get_accounts"):
        assert name in tools.READ_TOOL_NAMES, name
        assert name in _READ_EXECUTORS, name
        assert SOURCE_BY_TOOL[name].strip(), name
        # Neither is a propose tool: one prices a rule, the other explains access.
        assert name not in tools.PROPOSE_TOOL_NAMES, name


# --- estimate_restriction_cost ----------------------------------------------------
def test_pricing_a_restriction_writes_nothing_to_the_constraint_store() -> None:
    """The preview is the point of the surface, and a preview that saves is a bug."""
    title = _a_real_programme()
    before = _digest(CONSTRAINTS)
    payload = execute_read_tool("estimate_restriction_cost",
                                {"kind": "no_breaks", "where": _where(title)}, None)
    assert "error" not in payload, payload.get("error")
    assert payload["wrote_nothing"] is True
    assert _digest(CONSTRAINTS) == before, "pricing a restriction wrote a constraint row"


def test_the_two_money_bases_arrive_separately_and_are_never_summed() -> None:
    """Scored and exact answer different questions; blending them is the old lie."""
    title = _a_real_programme()
    payload = execute_read_tool("estimate_restriction_cost",
                                {"kind": "no_breaks", "where": _where(title)}, None)
    assert "scored" in payload and "exact" in payload
    scored, exact = payload["scored"], payload["exact"]
    # Each side answers for itself, including when it declines: an unavailable
    # basis states its reason rather than arriving as a zero or as the other one.
    for side in (scored, exact):
        assert "available" in side
        if not side["available"]:
            assert side.get("reason_he") or side.get("reason_en")
    if scored.get("available"):
        assert "revenue_delta" in scored and "basis" in scored
        # The delta is the product's own arithmetic, not re-derived here.
        assert scored["revenue_delta"] == pytest.approx(
            scored["revenue_after"] - scored["revenue_before"], abs=0.02)
    # No key blends the two. A total that summed them would be exactly the defect
    # the restriction surface's docstring says it exists to prevent.
    assert not {"total", "combined", "revenue_delta"} & set(payload)


def test_the_change_list_is_capped_with_its_true_total_beside_it() -> None:
    from kairos_api.assistant_read_tools_restriction import MAX_CHANGES

    title = _a_real_programme()
    payload = execute_read_tool("estimate_restriction_cost",
                                {"kind": "no_breaks", "where": _where(title)}, None)
    assert len(payload["changes"]) <= MAX_CHANGES
    assert payload["changes_total"] >= len(payload["changes"])
    if payload["changes_total"] > MAX_CHANGES:
        assert payload["changes_omitted"] == payload["changes_total"] - MAX_CHANGES


def test_a_kind_the_language_does_not_hold_is_refused_and_names_nothing_false() -> None:
    payload = execute_read_tool("estimate_restriction_cost", {"kind": "make_it_free"}, None)
    assert "error" in payload
    assert execute_read_tool("estimate_restriction_cost", {}, None)["error"]


def test_the_restriction_preview_stays_on_the_operators_own_channel() -> None:
    from kairos_api import channel_scope

    title = _a_real_programme()
    payload = execute_read_tool("estimate_restriction_cost",
                                {"kind": "no_breaks", "where": _where(title)}, None)
    assert payload["channel"] == channel_scope.operator_channel()


# --- get_accounts -----------------------------------------------------------------
def test_no_password_material_reaches_the_accounts_tool(monkeypatch) -> None:
    """The roster is built through the same projection the admin route serves.

    Scanning the whole payload for the word "password" is the guard that cannot
    tell code from a sentence about code: this tool's own prose says that
    resetting a password is a credential act it will not stage. So the check
    puts a real record carrying real credential fields through the real
    admin-visible path, and asserts those fields are the ones that do not
    survive it.
    """
    from kairos_api import auth, auth_store
    from kairos_api.assistant_read_tools_accounts import _roster

    secret_keys = ("password_scrypt", "password_salt", "session_token")
    planted = {"username": "orit.admin", "role": "admin", "affiliation": "company",
               "job": "trafficker", **{key: "SECRET-MATERIAL" for key in secret_keys}}
    monkeypatch.setattr(auth, "auth_active", lambda: True)
    monkeypatch.setattr(auth_store, "get_user", lambda username: planted)
    monkeypatch.setattr(auth_store, "load_users", lambda: [planted])

    roster = _roster("orit.admin")
    assert roster["available"] is True and roster["count"] == 1
    blob = json.dumps(roster, ensure_ascii=False, default=str)
    assert "SECRET-MATERIAL" not in blob
    for key in secret_keys:
        assert key not in blob, key
    # And the account is still identifiable, or the roster would be useless.
    assert roster["accounts"][0]["username"] == "orit.admin"
    assert roster["accounts"][0]["role"] == "admin"


def test_a_non_administrator_is_refused_the_roster_and_told_why(monkeypatch) -> None:
    """The role gate, exercised with authentication actually ON.

    With auth off there is no account to refuse, so the tool answers with the
    auth-off branch and the role gate is never reached; a test that ran only in
    that state asserted nothing about the wall it claims to check. So this one
    turns auth on and asks as an operator, which is a write role and still not
    an administrator.
    """
    from kairos_api import auth, auth_store
    from kairos_api.assistant_read_tools_accounts import ROSTER_ADMIN_ONLY_HE, _roster

    people = [{"username": "dana.ops", "role": "operator", "affiliation": "channel"},
              {"username": "orit.admin", "role": "admin", "affiliation": "company"}]
    monkeypatch.setattr(auth, "auth_active", lambda: True)
    monkeypatch.setattr(auth_store, "load_users", lambda: people)
    monkeypatch.setattr(auth_store, "get_user",
                        lambda username: next((p for p in people if p["username"] == username), None))

    for username in ("dana.ops", "nobody-at-all", ""):
        roster = _roster(username)
        assert roster["available"] is False, username
        assert roster["reason_he"] == ROSTER_ADMIN_ONLY_HE
        # An empty roster and a walled one are different facts, and one is a lie.
        assert "accounts" not in roster, username
    assert _roster("orit.admin")["available"] is True


def test_with_authentication_off_the_roster_says_that_rather_than_refusing() -> None:
    """Two different absences, and the tool must not report one as the other."""
    from kairos_api.assistant_read_tools_accounts import AUTH_OFF_HE

    roster = execute_read_tool("get_accounts", {}, None)["roster"]
    assert roster["available"] is False
    assert roster["reason_he"] == AUTH_OFF_HE
    assert "accounts" not in roster


def test_the_four_licence_limits_are_named_with_who_may_move_them() -> None:
    """The asymmetry this closes: Kai refused to change what it could not state."""
    from kairos_api import guardrail_store
    from kairos_api.assistant_permissions import guardrail_fields

    payload = execute_read_tool("get_accounts", {}, None)
    limits = payload["licence_limits"]
    assert tuple(limits["keys"]) == guardrail_store.GUARDRAIL_KEYS
    # Exactly the keys the propose path refuses on, so the tool that explains the
    # refusal and the code that makes it cannot name different fields.
    assert sorted(guardrail_fields({key: 1 for key in limits["keys"]})) == sorted(limits["keys"])
    assert limits["may_change_roles"] == ["admin"]
    assert set(limits["values"]) == set(guardrail_store.GUARDRAIL_KEYS)
    assert limits["effective_date"]


def test_the_roles_and_affiliations_come_from_the_stores_the_walls_consult() -> None:
    from kairos_api import auth_store
    from kairos_api.affiliation_wall import ADMIN_ROLES, WRITE_ROLES

    payload = execute_read_tool("get_accounts", {}, None)
    roles = {entry["role"]: entry for entry in payload["roles"]}
    assert set(roles) == set(auth_store.ROLES)
    for name, entry in roles.items():
        assert entry["may_change_anything_role_gates"] is (name in WRITE_ROLES)
        assert entry["manages_accounts"] is (name in ADMIN_ROLES)
        assert entry["note_he"], name
    assert {entry["affiliation"] for entry in payload["affiliations"]} == set(auth_store.AFFILIATIONS)


def test_the_tool_says_plainly_that_it_stages_no_account_change() -> None:
    """Credential acts are deliberately out of reach, and the tool says so."""
    payload = execute_read_tool("get_accounts", {}, None)
    assert "not available" in payload["proposing_account_changes"]
    for name in tools.PROPOSE_TOOL_NAMES:
        assert "account" not in name and "user" not in name and "password" not in name
