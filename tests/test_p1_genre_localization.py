"""Every programme genre in the plan has one Hebrew reading on recommendation titles."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"

PLAN_GENRES = (
    "Children", "Comedy", "Digital", "Documentary", "Drama", "Lifestyle",
    "Morning Program", "Music", "News", "Other", "Promo", "Reality",
    "Religious", "Special Event", "Talk Show",
)


def test_every_plan_genre_is_translated_inside_a_hebrew_server_sentence():
    if shutil.which("node") is None or not (DASHBOARD / "node_modules" / "vite").exists():
        pytest.skip("the dashboard runtime is unavailable")
    script = """
      import { runnerImport } from 'vite';
      const labels = await runnerImport('./src/shell/labels.js');
      const helper = await runnerImport('./src/shell/surface-helpers.js');
      const genres = JSON.parse(process.env.P1_GENRES);
      const rows = genres.map((genre) => ({
        genre,
        title: labels.module.localizedModelText(`בדיקת התוכנית ${genre} בשעה 20:00`, 'he'),
        label: labels.module.programTypeLabel(genre, 'he'),
        helper: helper.module.programTypeLabel(genre, 'he'),
      }));
      process.stdout.write(JSON.stringify(rows));
    """
    environment = dict(__import__("os").environ)
    environment["P1_GENRES"] = json.dumps(PLAN_GENRES)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=DASHBOARD,
        env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    for row in json.loads(result.stdout):
        assert row["genre"] not in row["title"], row
        assert row["label"] != row["genre"], row
        assert row["helper"] == row["label"], row
