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
SHELL = ROOT / "tv-break-dashboard" / "src" / "shell" / "styles.css"
TOKENS = ROOT / "tv-break-dashboard" / "src" / "tokens.css"


def _browser() -> str | None:
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        found = shutil.which(name)
        if found:
            return found
    mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    return str(mac) if mac.exists() else None


def _document() -> str:
    css = TOKENS.read_text(encoding="utf-8") + "\n" + SHELL.read_text(encoding="utf-8")
    return f"""<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><style>{css}</style></head>
<body><div class="kairos-shell rtl">
  <aside class="side-rail"><div class="brand-lockup"><strong>Kairos</strong></div></aside>
  <main class="workspace"><header class="top-bar">
    <div class="title-group"><span class="section-title">סקירה</span><button class="date-control">השבוע של 1 בנובמבר</button></div>
    <div class="command-group">
      <label class="scenario-select"><span>תרחיש</span><select><option>עדיפות להכנסות</option></select></label>
      <div class="risk-lambda-control"><div class="risk-lambda-head"><span class="risk-lambda-label">זהירות מסיכון</span><span>35/100</span></div><input type="range"></div>
      <button class="secondary-button">השוואת תרחישים</button>
    </div>
    <div class="status-group">
      <span class="api-state online">API חי</span><span class="freshness">עודכן 03:37</span>
      <button class="icon-button">R</button><button class="icon-button">K</button><button class="icon-button">N</button>
      <button class="secondary-button compact">English</button>
      <button class="run-button">הרצת אופטימיזציה</button>
      <button class="apply-button">החלה על לוח השידורים השבועי</button>
    </div>
  </header><section class="page-workspace">
    <div class="timeline-track">
      <button class="MuiButton-root timeline-break" id="narrow" style="left:0;width:20px"><span class="break-chip-clock">20:40</span><strong class="break-chip-detail">1/2</strong></button>
      <button class="MuiButton-root timeline-break" id="wide" style="left:40px;width:80px"><span class="break-chip-clock">20:40</span><strong class="break-chip-detail">1/2</strong></button>
    </div>
  </section></main>
</div><script>
requestAnimationFrame(() => requestAnimationFrame(() => {{
  const bar = document.querySelector('.top-bar');
  const controls = [...bar.querySelectorAll('button, select, .api-state, .freshness')];
  const result = {{
    documentFits: document.documentElement.scrollWidth <= window.innerWidth + 1,
    headerFits: bar.scrollWidth <= bar.clientWidth + 1,
    labelsFit: controls.every((node) => node.scrollWidth <= node.clientWidth + 1 && node.scrollHeight <= node.clientHeight + 1),
    labelsNoWrap: controls.every((node) => getComputedStyle(node).whiteSpace === 'nowrap'),
    headerWidth: bar.clientWidth,
    headerScrollWidth: bar.scrollWidth,
    headerHeight: bar.getBoundingClientRect().height,
    narrowMarkerIsClean: [...document.querySelectorAll('#narrow > *')].every((node) => getComputedStyle(node).display === 'none'),
    wideMarkerIsLegible: [...document.querySelectorAll('#wide > *')].every((node) => getComputedStyle(node).display !== 'none'),
  }};
  document.body.dataset.layout = JSON.stringify(result);
}}));
</script></body></html>"""


@pytest.mark.parametrize("width", [1600, 1500, 1280, 1000, 861, 860, 700])
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
    assert measured["labelsNoWrap"], (width, measured)
    assert measured["narrowMarkerIsClean"], (width, measured)
    assert measured["wideMarkerIsLegible"], (width, measured)
