"""Plan, week: publishing a plan version, diffing it and rolling it back.

Split out of ``week_api`` under the 450-line law, named by the helper rule, and
mounted on that module's router so the week keeps one registration.

Publishing is an internal freeze, per the owner ruling of 2026-08-01: the
planner names a version alone, the freeze records who and when, and the version
can be diffed against the one before it and restored byte for byte. It is not an
approval workflow and it is not a broadcast, so there is no second signature and
nothing leaves the building.

The permission is role-only, never affiliation: freezing a plan is the
operator's own act on the operator's own plan, so a channel account with a write
role performs it and a viewer reads the list and sees why the control is closed
before clicking it. Every read carries ``can_edit`` for exactly that reason.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from kairos_api import plan_version_store
from kairos_api.affiliation_wall import Wall, session_for

logger = logging.getLogger(__name__)

router = APIRouter(tags=["plan-versions"])

# Role-only: any affiliation may freeze the operator's plan, a viewer may not.
PUBLISH_WALL = Wall(
    detail="",
    company_only=False,
    role_detail="לחשבון צפייה אין הרשאה להקפיא גרסת תוכנית",
)


class PublishRequest(BaseModel):
    """The name a planner gives the frozen plan, and an optional note."""

    name: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=400)


def _actor(request: Optional[Request]) -> str:
    session = session_for(request)
    return str((session or {}).get("username") or "operator")


@router.get("/api/plan-versions")
def list_plan_versions(request: Request) -> dict[str, Any]:
    """Every frozen plan version, newest first, with the live plan's own state.

    ``live`` is what a freeze would capture right now, so a planner can see
    whether the plan on disk has moved since the last version before deciding to
    freeze another one.
    """
    manifests = plan_version_store.all_manifests()
    return PUBLISH_WALL.stamp(
        {"versions": manifests, "count": len(manifests), "live": plan_version_store.live_state()},
        request,
    )


@router.post("/api/plan-versions")
def publish_plan_version(payload: PublishRequest, request: Request) -> dict[str, Any]:
    """Freeze the saved plan under a name. The one write on this surface."""
    PUBLISH_WALL.require(request)
    try:
        manifest = plan_version_store.freeze(
            name=payload.name, actor=_actor(request), note=payload.note
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "There is no saved plan to freeze yet. Run the weekly plan first, "
                f"which writes {exc}."
            ),
        ) from exc
    return PUBLISH_WALL.stamp(dict(manifest), request)


@router.get("/api/plan-versions/{version_id}")
def read_plan_version(version_id: str, request: Request) -> dict[str, Any]:
    manifest = plan_version_store.get(version_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Unknown plan version {version_id}")
    return PUBLISH_WALL.stamp(dict(manifest), request)


@router.get("/api/plan-versions/{version_id}/diff")
def diff_plan_version(version_id: str, request: Request, against: Optional[str] = None) -> dict[str, Any]:
    """What this version changed, against its predecessor or against ``live``."""
    body = plan_version_store.diff(version_id, against=against)
    if not body.get("available") and str(body.get("reason", "")).startswith("no plan version"):
        raise HTTPException(status_code=404, detail=str(body["reason"]))
    return PUBLISH_WALL.stamp(body, request)


@router.post("/api/plan-versions/{version_id}/restore")
def restore_plan_version(version_id: str, request: Request) -> dict[str, Any]:
    """Put a frozen plan back, freezing the current one first so this is reversible."""
    PUBLISH_WALL.require(request)
    try:
        result = plan_version_store.restore(version_id, actor=_actor(request))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown plan version {version_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=f"That plan version has no frozen file: {exc}") from exc
    return PUBLISH_WALL.stamp(result, request)
