"""The affiliation wall and the ``can_edit`` contract, for any surface.

One mechanism, three uses, so a surface never invents its own rule:

- ``Wall.require(request)`` raises the 403 with its own Hebrew detail.
  ``Wall.require_read(request)`` is its read-side pair.
- ``Wall.guard()`` decorates a route so affiliation closes it on every method
  and role closes it on the write. FastAPI still injects the ``Request`` when
  the route does not declare one, because the wrapper republishes the signature
  with it, and a request body, a path parameter and a dependency all keep
  working, because the republished signature carries annotations already
  evaluated in the route's own module rather than strings FastAPI would try to
  evaluate in this one.
- ``Wall.stamp(payload, request)`` writes ``can_edit`` into any dict payload,
  with ``can_edit_reason`` carrying the exact detail the refusal would use, so
  a control renders as state before the click instead of failing after it.

The rule the walls encode is section 4.5 of the rebuild spec: affiliation
decides which side of the line an account can see, role decides what it can
change on its side. A wall therefore has two independent gates, and the reason
it reports names the gate that actually closed.

**Which gate applies to which method is the whole of that sentence.** Affiliation
is about seeing, so it closes a read and a write alike. Role is about changing,
so it closes only a write. The first round of this module enforced both on both,
and a wall declared the way the contract demonstrates it therefore answered 403
to a company **viewer** asking for a read, with the read-only-role refusal.
Measured before the fix, on real resolved sessions: a company viewer got 403 on
``GET`` through ``Wall(detail=..., company_only=True).guard()``, and 200 on
``POST`` through ``company_only(detail)``, which had no role gate at all. Both
are closed now, and both directions are pinned by
``tests/test_w0_cleanup_wall_reads.py``. A surface that genuinely needs a
role-gated read says ``guard(roles_on_read=True)``, so the strict form stays
reachable and stays visible at the call site.

Generalized from :mod:`kairos_api.events_access`, which walls the events
surface and the event pricing activation switch today. That module keeps its
own constants and its own call sites; this one delegates the affiliation
question straight to it, so there is exactly one implementation of "is this
requester company" in the process and the five existing call sites are
untouched.

An unresolvable request identity stays tolerant, exactly as the events guard already is: with
auth disabled, with no request object (direct in-process calls and bare-router
tests), or with no resolvable session (the server middleware already answers
401 before any route runs) the requester reads as company and as permitted. A
deployment without login keeps every surface open. A resolved account with a
channel or unresolved stored affiliation is walled; unresolved is the
fail-closed legacy migration state, not an unresolvable request identity.

Because unknown identity is tolerant, the guard must never mistake a parameter
for the request, or the tolerant path becomes a hole. It therefore finds the
route's ``Request`` by resolved **type** and never by parameter name, and it
injects its own under a reserved name. The name ``request`` is not reserved:
measured on the live app, 8 of the 115 routes call a request **body model**
``request``, all 8 of them writes. Being unsure resolves toward injecting, which
is always safe, so the failure direction is closed rather than open.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterable, Optional

from fastapi import HTTPException, Request

from kairos_api.events_access import (
    COMPANY_ONLY_DETAIL,
    EVENT_PRICING_COMPANY_ONLY_DETAIL,
    require_company_editor,
    requester_is_company,
)

# Roles that may change anything at all. Mirrors auth.WRITE_ROLES; duplicated as
# a frozenset here rather than imported so this module stays importable without
# the auth router, and a test pins the two together.
WRITE_ROLES = frozenset({"admin", "operator"})
ADMIN_ROLES = frozenset({"admin"})

# The methods that only read. A wall's role gate skips these, because role is
# the answer to "may this account change the thing" and a read changes nothing.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# The default refusals, in the product's first language.
COMPANY_SURFACE_DETAIL = "התצוגה הזו שמורה לצוות החברה"
READ_ONLY_ROLE_DETAIL = "לחשבון צפייה אין הרשאת עריכה"

__all__ = [
    "ADMIN_ROLES",
    "COMPANY_ONLY_DETAIL",
    "COMPANY_SURFACE_DETAIL",
    "EVENT_PRICING_COMPANY_ONLY_DETAIL",
    "READ_ONLY_ROLE_DETAIL",
    "SAFE_METHODS",
    "WRITE_ROLES",
    "Wall",
    "company_only",
    "has_role",
    "is_company",
    "require_company",
    "session_for",
]


def session_for(request: Optional[Request]) -> "dict[str, Any] | None":
    """The resolved session as ``{username, role, affiliation}``, or None.

    None means the identity is genuinely unknown (auth off, no request, or no
    live cookie), which every gate below reads as permitted for the reason in
    the module docstring. Never raises: an identity lookup failure must not
    turn a read into a 500.
    """
    from kairos_api import auth, auth_store

    if request is None or not auth.auth_active():
        return None
    try:
        session = auth_store.resolve_session(request.cookies.get(auth_store.COOKIE_NAME))
        if session is None:
            return None
        user = auth_store.get_user(session["username"]) or {}
        return {
            "username": session["username"],
            "role": str(session.get("role", "")),
            "affiliation": auth_store.normalize_affiliation(user.get("affiliation")),
        }
    except Exception:  # pragma: no cover - defensive, identity must not 500 a read
        return None


def is_company(request: Optional[Request]) -> bool:
    """Whether the requester is on the company side of the line."""
    return requester_is_company(request)


def require_company(request: Optional[Request], detail: str = COMPANY_ONLY_DETAIL) -> None:
    """Raise 403 with a Hebrew detail when a channel account crosses the line."""
    require_company_editor(request, detail=detail)


def has_role(request: Optional[Request], roles: Iterable[str]) -> bool:
    """Whether the session's role is in ``roles``, tolerant of unknown identity."""
    session = session_for(request)
    if session is None:
        return True
    return session["role"] in set(roles)


@dataclass(frozen=True)
class Wall:
    """One surface's permission rule, declared once and used on every route.

    ``detail`` is the refusal a channel account sees, ``role_detail`` the one a
    viewer sees. ``company_only`` false builds a role-only wall (any
    affiliation, a write role), which is what most run-side controls need.
    ``roles`` is the set that may change the thing; an empty set means the wall
    gates visibility only and nobody is refused on role.

    The two questions have separate answers. ``reason``, ``allows`` and
    ``require`` ask whether this account may CHANGE the thing, which is what
    ``stamp`` writes into ``can_edit``. Their ``_read`` counterparts ask whether
    it may SEE the thing, which consults affiliation and never role. ``guard``
    applies whichever one the request's method is subject to, so a route does not
    choose and cannot choose wrongly.
    """

    detail: str = COMPANY_SURFACE_DETAIL
    company_only: bool = True
    roles: frozenset[str] = WRITE_ROLES
    role_detail: str = READ_ONLY_ROLE_DETAIL

    def reason(self, request: Optional[Request]) -> Optional[str]:
        """Why this requester may not CHANGE the thing, or None when it may.

        Affiliation is checked first because it is the outer gate: a channel
        administrator is refused before their role is ever consulted. This is
        the string ``can_edit_reason`` carries.
        """
        blocked = self.read_reason(request)
        if blocked is not None:
            return blocked
        if self.roles and not has_role(request, self.roles):
            return self.role_detail
        return None

    def read_reason(self, request: Optional[Request]) -> Optional[str]:
        """Why this requester may not SEE the thing, or None when it may.

        Affiliation only. A viewer reads everything its side of the line holds,
        which is what makes a read-only account a usable account.
        """
        if self.company_only and not is_company(request):
            return self.detail
        return None

    def allows(self, request: Optional[Request]) -> bool:
        return self.reason(request) is None

    def allows_read(self, request: Optional[Request]) -> bool:
        return self.read_reason(request) is None

    def require(self, request: Optional[Request]) -> None:
        """Raise 403 with the reason that actually closed the gate, on a write."""
        _refuse(self.reason(request))

    def require_read(self, request: Optional[Request]) -> None:
        """The same, for a branch that only shows something."""
        _refuse(self.read_reason(request))

    def stamp(self, payload: "dict[str, Any]", request: Optional[Request]) -> "dict[str, Any]":
        """Write ``can_edit`` into a payload, in place, and return it.

        ``can_edit_reason`` is present only when the answer is false, and it is
        the same string the 403 would carry, so the refusal a person reads
        before the click and the one the server would send cannot drift.
        """
        reason = self.reason(request)
        payload["can_edit"] = reason is None
        if reason is None:
            payload.pop("can_edit_reason", None)
        else:
            payload["can_edit_reason"] = reason
        return payload

    def guard(
        self, *, roles_on_read: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorate a route: affiliation on every method, role on the write.

        ``roles_on_read`` restores the strict form, where the role gate closes
        a read too. A surface needs it only when seeing the thing is itself a
        privilege beyond seeing the side of the line it sits on.
        """

        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            return _wrap_with_wall(func, self, roles_on_read=roles_on_read)

        return decorate


def _refuse(reason: Optional[str]) -> None:
    if reason is not None:
        raise HTTPException(status_code=403, detail=reason)


def company_only(detail: str = COMPANY_SURFACE_DETAIL) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """The one-line form: a company-only route, read and write.

    It is the plain :meth:`Wall.guard` with the default roles, so its write is
    role-gated like every other wall's. It shipped with ``roles=frozenset()``,
    which meant a company viewer could POST through it; measured, and pinned in
    ``tests/test_w0_cleanup_wall_reads.py``.
    """
    return Wall(detail=detail, company_only=True).guard()


# ---------------------------------------------------------------------------
# The decorator plumbing: keep FastAPI's injection working
# ---------------------------------------------------------------------------

# The name the wrapper injects its own Request under. It is deliberately not
# ``request``: that name is already taken by a body model on 8 of the app's 115
# routes, and reusing it made ``Signature.replace`` raise "duplicate parameter
# name" on exactly the routes the wall most needs to close. Nothing but this
# module ever reads it, and the wrapper pops it before the route is called, so
# no route ever sees it.
_INJECTED_REQUEST = "kairos_wall_request"

# The postponed-annotation spellings of ``fastapi.Request``, matched whole so a
# body model called ``BreakDecisionRequest`` cannot be mistaken for one.
_REQUEST_ANNOTATIONS = frozenset({"Request", "fastapi.Request", "starlette.requests.Request"})


def _resolved_hints(func: Callable[..., Any]) -> "dict[str, Any]":
    """The route's annotations evaluated in the route module's own namespace.

    This is the load-bearing call of the whole decorator. The wrapper below is
    defined in this module, so ``wrapper.__globals__`` is this module, and
    FastAPI resolves a published signature's string annotations against the
    callable's globals. Under ``from __future__ import annotations``, which 79
    of the 80 modules in this package use, every annotation is a string, so a
    body model declared in the route's own module would not resolve here: it
    would fall through FastAPI's parameter analysis and become a query
    parameter, and the route would answer 422 on a body it was sent. Evaluating
    against ``func`` fixes the namespace once, before the signature is
    published, so no string is left for FastAPI to misresolve.

    ``include_extras`` keeps ``Annotated[..., Depends(...)]`` and the other
    FastAPI parameter markers intact, which is how a parameter carries its own
    metadata. An unresolvable annotation leaves the signature exactly as it is
    today rather than raising at import time.
    """
    import typing

    try:
        return typing.get_type_hints(func, include_extras=True)
    except Exception:  # pragma: no cover - measured zero of this package's 115 route endpoints
        return {}


def _request_parameter(signature: inspect.Signature, hints: "dict[str, Any]") -> Optional[str]:
    """The name of the route's own Request parameter, when it declares one.

    Only the resolved **type** decides. A parameter's name decides nothing,
    because the name ``request`` is not reserved: measured on the live app, 8 of
    the 115 routes call a request **body model** ``request``, and all 8 are
    writes (``POST /api/assistant/ask``, ``/api/assistant/ask/stream``,
    ``/api/scenario``, ``/api/optimizer-plan``, ``/api/optimal-plan``,
    ``/api/scenario-compare``, ``/api/jobs/recompute``, ``/api/break-decisions``).
    A wall that trusted the name took the declared branch on those routes, found
    a Pydantic model where it expected a ``Request``, read the identity as
    unknown and let everyone through. Returning None here is the fail-closed
    answer: the wrapper injects its own Request under a reserved name instead,
    and the wall resolves a real identity.

    Being unsure is safe in the same direction. When the type cannot be resolved
    the answer is None, an extra Request is injected under a name that cannot
    collide, and FastAPI fills the route's own Request parameter as it always
    did. The string branch below is the postponed-annotation form and only runs
    when hint resolution failed outright, measured zero of the 115 routes.
    """
    for name, parameter in signature.parameters.items():
        resolved = hints.get(name, parameter.annotation)
        if isinstance(resolved, type) and issubclass(resolved, Request):
            return name
        if isinstance(resolved, str) and resolved.strip() in _REQUEST_ANNOTATIONS:
            return name
    return None


def _typed_signature(signature: inspect.Signature, hints: "dict[str, Any]") -> inspect.Signature:
    """The same signature with every postponed annotation already evaluated.

    Only a string annotation is replaced, so a module that does not postpone
    its annotations gets back the signature it already had, unchanged.
    """
    parameters = [
        parameter.replace(annotation=hints[parameter.name])
        if isinstance(parameter.annotation, str) and parameter.name in hints
        else parameter
        for parameter in signature.parameters.values()
    ]
    returns = signature.return_annotation
    if isinstance(returns, str) and "return" in hints:
        returns = hints["return"]
    return signature.replace(parameters=parameters, return_annotation=returns)


def _exposed_signature(signature: inspect.Signature) -> inspect.Signature:
    """The signature FastAPI reads, with a Request appended for injection."""
    parameters = list(signature.parameters.values())
    injected = inspect.Parameter(
        _INJECTED_REQUEST, inspect.Parameter.KEYWORD_ONLY, annotation=Request
    )
    for index, parameter in enumerate(parameters):
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return signature.replace(parameters=parameters[:index] + [injected] + parameters[index:])
    return signature.replace(parameters=parameters + [injected])


def _wrap_with_wall(
    func: Callable[..., Any], wall: Wall, *, roles_on_read: bool = False,
) -> Callable[..., Any]:
    signature = inspect.signature(func)
    hints = _resolved_hints(func)
    own = _request_parameter(signature, hints)
    typed = _typed_signature(signature, hints)
    exposed = typed if own else _exposed_signature(typed)

    def _request_from(args: tuple, kwargs: dict) -> Optional[Request]:
        if own:
            bound = exposed.bind_partial(*args, **kwargs)
            value = bound.arguments.get(own)
            return value if isinstance(value, Request) else None
        value = kwargs.pop(_INJECTED_REQUEST, None)
        return value if isinstance(value, Request) else None

    def _close(request: Optional[Request]) -> None:
        """Apply the gates this method is subject to.

        An unreadable method reads as a write, which is the closed direction:
        the role gate applies, and unknown identity stays permitted by the same
        tolerance the whole module documents.
        """
        method = str(getattr(request, "method", "") or "").upper()
        if roles_on_read or method not in SAFE_METHODS:
            wall.require(request)
        else:
            wall.require_read(request)

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            _close(_request_from(args, kwargs))
            return await func(*args, **kwargs)

        async_wrapper.__signature__ = exposed  # type: ignore[attr-defined]
        async_wrapper.kairos_wall = wall  # type: ignore[attr-defined]
        return async_wrapper

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _close(_request_from(args, kwargs))
        return func(*args, **kwargs)

    wrapper.__signature__ = exposed  # type: ignore[attr-defined]
    wrapper.kairos_wall = wall  # type: ignore[attr-defined]
    return wrapper
