"""The assistant's visible identity is one deliberate product concept.

The implementation keeps the historic internal ``kai`` module/API names for
compatibility, while every operator-facing entrance uses Mabat and the same
calibrated watch-mark.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "tv-break-dashboard" / "src"


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_mabat_is_the_single_visible_assistant_name() -> None:
    vocabulary = source("vocabulary.js")
    assert "en: 'Mabat'" in vocabulary
    assert "he: 'מבט'" in vocabulary

    live = "\n".join(
        source(path)
        for path in (
            "shell/top-bar.jsx",
            "shell/side-rail.jsx",
            "kai/AssistantDock.jsx",
            "kai/AssistantPanel.jsx",
            "kai/AssistantComposer.jsx",
        )
    )
    assert "Mabat" in live
    assert "מבט" in live
    assert "'Kai, the AI assistant'" not in live
    assert "'קאי, עוזר" not in live


def test_the_watch_mark_is_shared_by_every_live_entrance() -> None:
    icons = source("shell/kairos-icons.jsx")
    assert "export const MabatIcon = iconComponent('mabat')" in icons
    assert 'M3.4 12c2.5-3.8' in icons
    assert "MabatIcon" in source("shell/top-bar.jsx")
    assert "MabatIcon" in source("shell/side-rail.jsx")
    assert "MabatIcon" in source("kai/AssistantDock.jsx")


def test_the_dock_announcement_is_hidden_and_provider_internals_stay_off_screen() -> None:
    dock = source("kai/AssistantDock.jsx")
    studio = source("studio/studio.css")
    panel = source("kai/AssistantPanel.jsx")
    assert 'className="studio-visually-hidden"' in dock
    assert ".studio-visually-hidden" in studio
    assert "clip-path: inset(50%)" in studio
    assert "ANTHROPIC_API_KEY" not in panel
    assert "KAIROS_ASSISTANT_API_KEY" not in panel
