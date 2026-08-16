"""Focused source contracts for the pre-workspace authentication boundary."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "tv-break-dashboard" / "src" / "shell"
FRONTEND = ROOT / "tv-break-dashboard"


def read(name: str) -> str:
    return (SHELL / name).read_text(encoding="utf-8")


def function_block(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {signature}")


def contrast(first: str, second: str) -> float:
    def luminance(value: str) -> float:
        channels = [int(value[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


@pytest.fixture(scope="module")
def auth_guard_decisions() -> dict[str, bool]:
    script = r"""
      import { createServer } from 'vite';
      const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
      try {
        const integrity = await server.ssrLoadModule('/src/shell/auth-integrity.js');
        process.stdout.write(JSON.stringify({
          open: integrity.workspaceSessionReady({ status: 'open', user: null }),
          ready: integrity.workspaceSessionReady({ status: 'ready', user: { username: 'operator' } }),
          missingUser: integrity.workspaceSessionReady({ status: 'ready', user: null }),
          unknown: integrity.workspaceSessionReady({ status: 'future-state', user: { username: 'operator' } }),
          users401: integrity.shouldExpireSession('/api/auth/users', 401),
          password401: integrity.shouldExpireSession('/api/auth/change-password', 401),
          job401: integrity.shouldExpireSession('https://kairos.test/api/auth/job', 401),
          session401: integrity.shouldExpireSession('/api/auth/session', 401),
          login401: integrity.shouldExpireSession('/api/auth/login', 401),
          logout401: integrity.shouldExpireSession('/api/auth/logout', 401),
          lookalike401: integrity.shouldExpireSession('/api/auth/login-extra', 401),
          users403: integrity.shouldExpireSession('/api/auth/users', 403),
        }));
      } finally { await server.close(); }
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-"],
        cwd=FRONTEND,
        input=script,
        text=True,
        check=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def test_session_probe_errors_fail_closed_before_any_workspace_mount(auth_guard_decisions) -> None:
    session = read("use-session.js")
    assert "result.status === 0 ? 'offline' : result.status === 503 ? 'setup' : ''" in session
    assert "result.data.auth_disabled === true" in session
    assert "result.data.authenticated === true && result.data.username" in session
    assert session.count("setAuth({ status: 'open', user: null })") == 1
    open_branch = session.index("result.data.auth_disabled")
    error_branch = session.index("result.status === 0 || result.status === 503")
    assert open_branch < session.index("setAuth({ status: 'open', user: null })") < error_branch
    assert "setAuth({ status: 'login', user: null })" in session[error_branch:]

    login = read("Login.jsx")
    assert "value === 'offline' || value === 'setup'" in login
    assert "scripts/init_auth.py" in login
    assert "window.location.reload()" in login
    assert "סביבת העבודה נשארת נעולה" in login

    auth_screens = read("auth-screens.jsx")
    app = read("App.jsx")
    assert "if (!workspaceSessionReady(auth))" in auth_screens
    gate = app.index("if (authScreen) return authScreen;")
    workspace = app.index("return <TVBreakDashboard")
    assert gate < workspace
    assert auth_guard_decisions["open"] is True
    assert auth_guard_decisions["ready"] is True
    assert auth_guard_decisions["missingUser"] is False
    assert auth_guard_decisions["unknown"] is False


def test_only_bootstrap_login_and_logout_401_are_exempt(auth_guard_decisions) -> None:
    assert auth_guard_decisions == {
        "open": True,
        "ready": True,
        "missingUser": False,
        "unknown": False,
        "users401": True,
        "password401": True,
        "job401": True,
        "session401": False,
        "login401": False,
        "logout401": False,
        "lookalike401": True,
        "users403": False,
    }
    session = read("auth-integrity.js")
    assert "!url.includes('/api/auth/')" not in session
    assert "AUTH_401_EXEMPT_PATHS.has(path)" in session


def test_logout_changes_auth_state_only_after_success() -> None:
    block = function_block(read("TVBreakDashboard.jsx"), "async function handleLogout")
    request = block.index("const result = await requestLogout()")
    failure = block.index("if (!result.ok)")
    success = block.index("notify('Signed out.'")
    transition = block.index("setAuth({ status: 'login', user: null })")
    assert request < failure < success < transition
    assert "Your session is still active; try again." in block
    assert "return;" in block[failure:success]


def test_account_modals_use_the_canonical_native_dialog_contract() -> None:
    password = read("Login.jsx")
    accounts = read("UserAdminDialog.jsx")
    modal = read("modal-primitives.jsx")
    css = read("login.css")

    for source in (password, accounts):
        assert "import { Dialog } from '../studio/modal';" in source
        assert "<Dialog" in source
        assert 'role="dialog"' not in source
        assert 'aria-modal="true"' not in source

    assert "initialFocusRef={currentRef}" in password
    assert "dismissOnBackdrop={!forced}" in password
    assert "onClose={forced ? undefined : onClose}" in password
    assert "event.key === 'Escape'" in password
    assert "if (event.key === 'Escape') event.preventDefault();" in password
    assert ".auth-dialog-forced .studio-modal__close { display: none; }" in css

    assert "<dialog" in modal
    assert "dialog.showModal()" in modal
    assert "useFocusReturn(open)" in modal
    assert "event.preventDefault(); onClose?.('escape')" in modal
    assert "aria-labelledby={titleId}" in modal


def test_tooltip_and_auth_primary_foregrounds_meet_text_contrast() -> None:
    theme = read("theme.js")
    css = read("login.css")
    palette = dict(re.findall(r"^\s+(\w+): '(#[0-9a-f]{6})',?$", theme, re.MULTILINE))
    tooltip = theme[theme.index("MuiTooltip:") : theme.index("MuiPopover:")]
    primary = re.search(r"\.login-submit,\s*\.auth-primary\s*\{([^}]+)\}", css, re.DOTALL)

    assert "backgroundColor: studioPalette.chrome" in tooltip
    assert "color: studioPalette.surface" in tooltip
    assert primary and "background: var(--accent-strong);" in primary.group(1)
    assert "color: var(--surface);" in primary.group(1)
    assert contrast(palette["surface"], palette["chrome"]) >= 4.5
    assert contrast(palette["surface"], palette["accentStrong"]) >= 4.5


def test_owned_frontend_files_stay_within_the_source_cap() -> None:
    for name in ("auth-integrity.js", "use-session.js", "Login.jsx", "UserAdminDialog.jsx", "TVBreakDashboard.jsx", "login.css"):
        assert len(read(name).splitlines()) < 450, name
    # Existing theme module is exactly at the legacy ceiling and was outside
    # this safety pass; do not let it grow while the dedicated split lands.
    assert len(read("theme.js").splitlines()) <= 450
