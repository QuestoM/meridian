"""Wave 3 header proof in a real browser layout engine.

The source guard catches a deleted rule. This probe catches the different
failure the owner reported: all rules exist, yet the max-content groups are
wider than the workspace and a button's label escapes its border.
"""

from __future__ import annotations

import html
import json
import os
import re
import signal
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "tv-break-dashboard"
INDEX = (DASH / "src" / "index.jsx").read_text(encoding="utf-8")
GLOBAL_SHEETS = tuple(
    DASH / "src" / specifier[2:]
    for specifier in re.findall(r"import '([^']+\.css)';", INDEX)
    if specifier.startswith("./")
)
LAYOUT_SHEETS = tuple(
    path for path in GLOBAL_SHEETS
    if path.relative_to(DASH / "src").as_posix() in {
        "tokens.css",
        "shell/styles.css",
        "shell/styles-timeline.css",
        "shell/styles-workspaces.css",
        "studio/studio.css",
        "studio/studio-workspaces.css",
        "shell/studio-shell.css",
    }
)


def _browser() -> str | None:
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        found = shutil.which(name)
        if found:
            return found
    mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    return str(mac) if mac.exists() else None


def _document() -> str:
    css = "\n".join(path.read_text(encoding="utf-8") for path in LAYOUT_SHEETS)
    return f"""<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><style>{css}</style></head>
<body><div class="kairos-shell rtl">
  <aside class="side-rail"><div class="brand-lockup"><strong>Kairos</strong></div></aside>
  <main class="workspace"><header class="top-bar">
    <div class="top-bar-primary">
      <div class="title-group"><span class="section-title">ממשל וכללים</span><button class="MuiButton-root date-control">השבוע של 1 בנובמבר</button></div>
      <div class="status-group">
        <div class="connection-state"><span class="api-state online">API חי</span><span class="freshness">עודכן 03:37</span></div>
        <button class="MuiButton-root locale-toggle"><span class="locale-toggle-label">English</span></button>
        <button class="MuiButton-root icon-button" aria-label="פעילות">N</button>
        <button class="MuiButton-root icon-button" aria-label="קאי">K</button>
      </div>
    </div>
    <nav class="context-local-nav"><button class="active">הגבלות</button><button>רישיון</button><button>מחירון</button><button>לוח אירועים</button><button>ערוץ ומודל</button></nav>
  </header><section class="page-workspace">
    <div class="timeline-track">
      <button class="MuiButton-root timeline-break" id="narrow" style="left:0;width:20px"><span class="break-chip-clock">20:40</span><strong class="break-chip-detail">1/2</strong></button>
      <button class="MuiButton-root timeline-break" id="wide" style="left:40px;width:80px"><span class="break-chip-clock">20:40</span><strong class="break-chip-detail">1/2</strong></button>
    </div>
  </section></main>
</div><script>
(() => {{
  const bar = document.querySelector('.top-bar');
  const controls = [...bar.querySelectorAll('button, .api-state, .freshness')];
  const result = {{
    documentFits: document.documentElement.scrollWidth <= window.innerWidth + 1,
    headerFits: bar.scrollWidth <= bar.clientWidth + 1,
    labelsFit: controls.every((node) => node.scrollWidth <= node.clientWidth + 1 && node.scrollHeight <= node.clientHeight + 1),
    headerWidth: bar.clientWidth,
    headerScrollWidth: bar.scrollWidth,
    headerHeight: bar.getBoundingClientRect().height,
    narrowMarkerIsClean: [...document.querySelectorAll('#narrow > *')].every((node) => getComputedStyle(node).display === 'none'),
    wideMarkerIsLegible: [...document.querySelectorAll('#wide > *')].every((node) => getComputedStyle(node).display !== 'none'),
  }};
  document.body.dataset.layout = JSON.stringify(result);
}})();
</script></body></html>"""


@pytest.mark.parametrize("width", [1600, 1500, 1400, 1399, 1280, 1200])
def test_header_has_no_real_overflow_at_supported_widths(tmp_path: Path, width: int) -> None:
    browser = _browser()
    if browser is None:
        pytest.skip("no Chromium-family browser is installed for the layout proof")
    page = tmp_path / "header.html"
    page.write_text(_document(), encoding="utf-8")
    command = [
        browser,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-background-networking",
        "--disable-component-update",
        "--no-first-run",
        "--no-default-browser-check",
        "--hide-scrollbars",
        "--virtual-time-budget=1000",
        f"--user-data-dir={tmp_path / 'chrome-profile'}",
        f"--window-size={width},900",
        "--dump-dom",
        page.as_uri(),
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=8)
    except subprocess.TimeoutExpired as exc:
        # Some macOS Chrome builds leave their updater helper alive after
        # --dump-dom has already emitted the measured page. Terminate that
        # helper and use the complete DOM only; a missing measurement still
        # fails below, so this cannot turn an unmeasured layout into a pass.
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    if process.returncode == -6 and not stdout and not stderr:
        pytest.skip("the local Chrome binary is blocked by the command sandbox")
    if not timed_out:
        assert process.returncode == 0, stderr[-1000:]
    match = re.search(r'data-layout="([^"]+)"', stdout)
    assert match, stdout[-1000:] or stderr[-1000:]
    measured = json.loads(html.unescape(match.group(1)))
    assert measured["documentFits"], (width, measured)
    assert measured["headerFits"], (width, measured)
    assert measured["labelsFit"], (width, measured)
    assert measured["narrowMarkerIsClean"], (width, measured)
    assert measured["wideMarkerIsLegible"], (width, measured)
