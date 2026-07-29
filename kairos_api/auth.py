"""Login, sessions and role enforcement for the Kairos API.

Lifecycle (one client per deployment, multiple named accounts per client):

- Fresh clone: data/auth/users.json does not exist, so authentication is OFF.
  Every /api route behaves exactly as before, GET /api/auth/me answers
  {"auth_disabled": true} and the dashboard renders without a login wall,
  showing an honest "login not set up" state instead of a fake identity.
- Seeding: the operator runs scripts/init_auth.py (prints a one-time admin
  password, also written once to data/auth/initial_admin_password.txt with
  mode 600), or sets KAIROS_ADMIN_PASSWORD before startup, which seeds the
  admin account on boot. KAIROS_AUTH_DIR relocates the store (tests use it).
- Seeded: enforcement is ON immediately, even on a running server, because the
  guard checks store existence per request. Every /api route requires a live
  session except POST /api/auth/login and GET /api/health. Mutating methods
  (POST/PUT/PATCH/DELETE) outside /api/auth/ additionally require the operator
  or admin role, so a viewer session is read-only. /api/auth/users* requires
  admin. Non-API routes (the built SPA and its assets) stay open so the app
  shell can load and show the login screen.
- Escape hatch: KAIROS_AUTH_DISABLED=1 bypasses enforcement entirely and is
  logged loudly at startup. Meant for tests and local development only.

Sessions are opaque secrets.token_urlsafe(32) values held in an in-process
dict with a 12 hour sliding expiry, which is correct for the single-process
uvicorn deployment. The kairos_session cookie is HttpOnly, SameSite=Lax,
Path=/ and intentionally has no Max-Age (the server enforces expiry). The
Secure flag defaults off so plain-HTTP localhost works; a TLS deployment
sets KAIROS_COOKIE_SECURE=1 so the browser never sends the session cookie
over plain HTTP.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from kairos_api import auth_store as store

logger = logging.getLogger("kairos.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])

PUBLIC_API_PATHS = frozenset({"/api/auth/login", "/api/health"})
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
WRITE_ROLES = frozenset({"admin", "operator"})


def auth_bypassed() -> bool:
    return os.getenv("KAIROS_AUTH_DISABLED", "").strip().lower() in {"1", "true", "yes"}


def _cookie_secure() -> bool:
    """Send the session cookie with the Secure flag when KAIROS_COOKIE_SECURE is
    set (a TLS deployment). Default off so plain-HTTP localhost keeps working."""
    return os.getenv("KAIROS_COOKIE_SECURE", "").strip().lower() in {"1", "true", "yes"}


def _activity(event: str, user: str, role: str = "") -> None:
    """Best-effort activity log hook for login, login_failed and logout.

    The activity log module is imported lazily to avoid a module cycle, and
    the call is wrapped so a logging failure can never break the sign-in
    flow. Only the event name, the username and the role are recorded here;
    the password never reaches the logger.
    """
    try:
        from kairos_api import activity_log

        activity_log.record_auth_event(event, user=user, role=role)
    except Exception:
        logger.debug("activity log hook failed for %s", event, exc_info=True)


def auth_active() -> bool:
    """Enforcement is on whenever the store is seeded and the escape hatch is unset."""
    return not auth_bypassed() and store.store_initialized()


def _session_from_request(request: Request) -> dict[str, Any] | None:
    return store.resolve_session(request.cookies.get(store.COOKIE_NAME))


def enforce_request(request: Request) -> JSONResponse | None:
    """The single enforcement rule, called by the middleware in server.py.

    Returns None to let the request through, or a JSONResponse denial.
    Resolving the session here also slides its expiry forward on every use.
    """
    path = request.url.path
    if not path.startswith("/api/"):
        return None  # static shell and SPA assets stay open
    if not auth_active():
        return None
    if path in PUBLIC_API_PATHS or request.method == "OPTIONS":
        return None
    session = _session_from_request(request)
    if session is None:
        return JSONResponse(status_code=401, content={"detail": "A signed-in session is required."})
    if path.startswith("/api/auth/users") and session["role"] != "admin":
        return JSONResponse(status_code=403, content={"detail": "The admin role is required for account management."})
    if (
        request.method in MUTATING_METHODS
        and not path.startswith("/api/auth/")
        and session["role"] not in WRITE_ROLES
    ):
        return JSONResponse(status_code=403, content={"detail": "A viewer session is read-only."})
    return None


@router.on_event("startup")
def _announce_auth_state() -> None:
    if auth_bypassed():
        logger.warning(
            "KAIROS_AUTH_DISABLED is set: API authentication is bypassed for every request. "
            "Never run a real deployment this way."
        )
        return
    if not store.store_initialized():
        if os.getenv("KAIROS_ADMIN_PASSWORD"):
            try:
                store.seed_initial_admin()
                logger.info("Auth store seeded on startup from KAIROS_ADMIN_PASSWORD; login is now required.")
            except (RuntimeError, ValueError) as exc:
                logger.error("Could not seed the auth store from KAIROS_ADMIN_PASSWORD: %s", exc)
        else:
            logger.info(
                "Auth store not initialized (%s missing): the API is open. "
                "Run scripts/init_auth.py to enable login.",
                store.users_path(),
            )
    else:
        logger.info("Auth store initialized: API requests require a signed-in session.")


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=store.MIN_PASSWORD_LENGTH)


class CreateUserRequest(BaseModel):
    username: str
    password: str = Field(min_length=store.MIN_PASSWORD_LENGTH)
    role: Literal["admin", "operator", "viewer"]
    display_name: str = ""
    # Admin-created accounts get a temporary password by default, so the
    # dashboard forces a password change at the first sign-in.
    must_change_password: bool = True
    # Company staff manage everything; a channel-affiliated account cannot
    # manage calendar events or the event pricing activation switch.
    affiliation: Literal["company", "channel"] = "company"


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(min_length=store.MIN_PASSWORD_LENGTH)


class AffiliationRequest(BaseModel):
    affiliation: Literal["company", "channel"]


def _public_user(record: dict[str, Any]) -> dict[str, Any]:
    """The over-the-wire shape: never includes password material."""
    return {
        "username": record.get("username", ""),
        "display_name": record.get("display_name", "") or record.get("username", ""),
        "role": record.get("role", ""),
        "created_at": record.get("created_at", ""),
        "must_change_password": bool(record.get("must_change_password", False)),
        "affiliation": store.normalize_affiliation(record.get("affiliation")),
    }


def _require_session(request: Request) -> dict[str, Any]:
    session = _session_from_request(request)
    if session is None:
        raise HTTPException(status_code=401, detail="A signed-in session is required.")
    return session


def _require_admin(request: Request) -> dict[str, Any]:
    session = _require_session(request)
    if session["role"] != "admin":
        raise HTTPException(status_code=403, detail="The admin role is required for account management.")
    return session


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@router.post("/login")
def login(payload: LoginRequest, response: Response) -> dict[str, Any]:
    if not store.store_initialized():
        raise HTTPException(
            status_code=503,
            detail="Login is not set up yet. Run scripts/init_auth.py on the server first.",
        )
    username = store.normalize_username(payload.username)
    retry_after = store.login_retry_after(username)
    if retry_after > 0:
        raise HTTPException(
            status_code=429,
            detail="Too many failed sign-in attempts for this username. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )
    user = store.get_user(username)
    if user is None:
        store.burn_password_check(payload.password)
        store.record_login_failure(username)
        _activity("login_failed", username)
        raise HTTPException(status_code=401, detail="The username or password is incorrect.")
    if not store.verify_password(payload.password, user.get("password_scrypt") or {}):
        store.record_login_failure(username)
        _activity("login_failed", username)
        raise HTTPException(status_code=401, detail="The username or password is incorrect.")
    store.clear_login_failures(username)
    token = store.create_session(username, user.get("role", "viewer"))
    _activity("login", username, user.get("role", "viewer"))
    # Secure only when KAIROS_COOKIE_SECURE says the deployment sits behind TLS
    # (default off for plain-HTTP localhost). No Max-Age: the server enforces
    # the 12h sliding expiry.
    response.set_cookie(store.COOKIE_NAME, token, httponly=True, samesite="lax", path="/",
                        secure=_cookie_secure())
    return _public_user(user)


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, Any]:
    session = _session_from_request(request)
    store.drop_session(request.cookies.get(store.COOKIE_NAME))
    response.delete_cookie(store.COOKIE_NAME, path="/", secure=_cookie_secure())
    if session is not None:
        _activity("logout", session["username"], session.get("role", ""))
    return {"signed_out": True}


@router.get("/me")
def me(request: Request) -> dict[str, Any]:
    if not auth_active():
        # Uninitialized store or explicit bypass: tell the dashboard, honestly,
        # that there is no login wall instead of pretending someone signed in.
        return {"auth_disabled": True}
    session = _require_session(request)
    user = store.get_user(session["username"])
    if user is None:
        store.drop_session(request.cookies.get(store.COOKIE_NAME))
        raise HTTPException(status_code=401, detail="This account no longer exists.")
    return {"auth_disabled": False, **_public_user(user)}


@router.post("/change-password")
def change_password(payload: ChangePasswordRequest, request: Request) -> dict[str, Any]:
    session = _require_session(request)
    user = store.get_user(session["username"])
    if user is None:
        raise HTTPException(status_code=401, detail="This account no longer exists.")
    if not store.verify_password(payload.current_password, user.get("password_scrypt") or {}):
        raise HTTPException(status_code=403, detail="The current password is incorrect.")
    try:
        updated = store.set_password(session["username"], payload.new_password, must_change_password=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _public_user(updated)


# ---------------------------------------------------------------------------
# Account management (admin only; the middleware also guards /api/auth/users*)
# ---------------------------------------------------------------------------

@router.get("/users")
def list_users(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return {"users": [_public_user(user) for user in store.load_users()]}


@router.post("/users", status_code=201)
def create_user(payload: CreateUserRequest, request: Request) -> dict[str, Any]:
    _require_admin(request)
    try:
        record = store.add_user(
            payload.username,
            payload.password,
            payload.role,
            payload.display_name,
            must_change_password=payload.must_change_password,
            affiliation=payload.affiliation,
        )
    except store.DuplicateUserError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _public_user(record)


@router.delete("/users/{username}")
def delete_user(username: str, request: Request) -> dict[str, Any]:
    session = _require_admin(request)
    username = store.normalize_username(username)
    target = store.get_user(username)
    if target is None:
        raise HTTPException(status_code=404, detail=f"No account named {username}.")
    if target.get("role") == "admin" and store.admin_count() <= 1:
        raise HTTPException(status_code=400, detail="The last admin account cannot be deleted.")
    if username == session["username"]:
        raise HTTPException(status_code=400, detail="You cannot delete the account you are signed in with.")
    store.remove_user(username)
    store.drop_sessions_for(username)
    return {"deleted": username}


@router.put("/users/{username}/affiliation")
def set_affiliation(username: str, payload: AffiliationRequest, request: Request) -> dict[str, Any]:
    """Flip an account between company and channel affiliation (admin only).

    Takes effect immediately: the events write guard and the event pricing
    activation guard read the store per request, so live sessions do not need
    to sign in again to gain or lose the company surface.
    """
    _require_admin(request)
    username = store.normalize_username(username)
    if store.get_user(username) is None:
        raise HTTPException(status_code=404, detail=f"No account named {username}.")
    updated = store.set_affiliation(username, payload.affiliation)
    return _public_user(updated)


@router.post("/users/{username}/reset-password")
def reset_password(username: str, payload: ResetPasswordRequest, request: Request) -> dict[str, Any]:
    _require_admin(request)
    username = store.normalize_username(username)
    if store.get_user(username) is None:
        raise HTTPException(status_code=404, detail=f"No account named {username}.")
    updated = store.set_password(username, payload.new_password, must_change_password=True)
    # A reset invalidates the person's live sessions; they sign in again with
    # the temporary password and are forced to replace it.
    store.drop_sessions_for(username)
    return _public_user(updated)
