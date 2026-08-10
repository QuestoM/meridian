"""P13: real metadata joins by House Number and measured failures stop locking."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi import HTTPException

from kairos_api import break_api_pod, media_verdict
from kairos_api.media_ingest import MediaReportError, import_report
from kairos_api.media_store import ASSETS_PATH, COLUMNS, FAILED, UNAVAILABLE, VERIFIED, read_assets, write_assets
from kairos_api.media_verdict import lock_refusal, verdict_for, verdicts_for


def _rules(**over):
    rules = {
        "duration_tolerance_seconds": 0.04,
        "accepted_container_formats": ["mxf"],
        "accepted_video_codecs": ["xdcamhd422"],
        "accepted_frame_rates": ["25.0"],
        "accepted_display_aspect_ratios": ["16:9"],
        "accepted_pixel_dimensions": ["1920x1080"],
        "required_audio": True,
        "accepted_audio_channel_layouts": ["stereo"],
        "accepted_loudness_standards": ["owner-r128"],
        "loudness_target_lufs": -23.0,
        "loudness_tolerance_lu": 1.0,
        "approved_states": ["approved"],
        "rejected_states": ["rejected"],
        "source": "owner playout standard",
        "configured": True,
    }
    rules.update(over)
    return rules


def _asset(**over):
    row = {
        "house_number": "H123",
        "duration_seconds": 30.0,
        "duration_frames": 750,
        "frame_rate": 25.0,
        "container_format": "mxf",
        "video_codec": "xdcamhd422",
        "pixel_width": 1920,
        "pixel_height": 1080,
        "display_aspect_ratio": "16:9",
        "audio_present": True,
        "audio_channel_layout": "stereo",
        "loudness_lufs": -23.0,
        "loudness_standard": "owner-r128",
        "approval_state": "approved",
        "approval_authority": "QC",
        "approved_at": "2026-08-10T00:01:00Z",
        "measured_at": "2026-08-10T00:00:00Z",
        "source": "qc-report.csv",
    }
    row.update(over)
    return row


def test_shipped_store_and_standards_are_honestly_empty() -> None:
    assert ASSETS_PATH.exists() and read_assets() == []
    assert ASSETS_PATH.read_text(encoding="utf-8-sig").splitlines()[0].split(",") == list(COLUMNS)
    from kairos_api.media_standards import load_standards
    assert load_standards()["configured"] is False


def test_a_complete_measured_asset_verifies() -> None:
    result = verdict_for("H123", 30.0, {"H123": _asset()}, _rules())
    assert result["state"] == VERIFIED and result["blocks_lock"] is False
    assert set(result["facts"]) == {"duration", "container", "codec", "frame_rate", "frame_shape", "audio", "loudness", "approval"}
    assert {fact["state"] for fact in result["facts"].values()} == {VERIFIED}


@pytest.mark.parametrize(
    "over,broken",
    [
        ({"duration_seconds": 29.0}, "duration"),
        ({"duration_frames": 700}, "duration"),
        ({"container_format": "avi"}, "container"),
        ({"video_codec": "h264"}, "codec"),
        ({"frame_rate": 50.0, "duration_frames": 1500}, "frame_rate"),
        ({"pixel_width": 720}, "frame_shape"),
        ({"audio_present": False}, "audio"),
        ({"loudness_lufs": -18.0}, "loudness"),
        ({"approval_state": "rejected"}, "approval"),
    ],
)
def test_each_measured_failure_fires_independently_and_blocks(over, broken) -> None:
    result = verdict_for("H123", 30.0, {"H123": _asset(**over)}, _rules())
    assert result["state"] == FAILED and result["blocks_lock"] is True
    assert result["facts"][broken]["state"] == FAILED
    assert all(fact["state"] == VERIFIED for key, fact in result["facts"].items() if key != broken)


def test_missing_measurement_or_owner_standard_is_unavailable_not_clean() -> None:
    no_asset = verdict_for("H404", 30.0, {}, _rules())
    no_standard = verdict_for("H123", 30.0, {"H123": _asset()}, _rules(accepted_video_codecs=[]))
    assert no_asset["state"] == no_standard["state"] == UNAVAILABLE
    assert no_asset["blocks_lock"] is no_standard["blocks_lock"] is False
    assert no_standard["facts"]["codec"]["state"] == UNAVAILABLE


def test_pod_join_uses_house_number_not_the_version_name(monkeypatch) -> None:
    monkeypatch.setattr(media_verdict, "assets_by_house_number", lambda: {"H123": _asset()})
    monkeypatch.setattr(media_verdict, "load_standards", _rules)
    spot = {"creative": {"value": "A version name"}, "house_number": {"value": "H123"}, "duration": {"seconds": 30.0}}
    result = verdicts_for([spot])
    assert result["counts"][VERIFIED] == 1
    assert result["spots"][0]["house_number"] == "H123"


def test_failed_media_is_a_server_side_lock_refusal(monkeypatch) -> None:
    from kairos_api import break_api_pod_order

    pod = {"media": {"blocks_lock": True, "blocking_house_numbers": ["H123"]},
           "order": {"locked": False}, "fingerprint": "fp"}
    monkeypatch.setattr(break_api_pod, "_pod_or_404", lambda _pod_id: pod)
    monkeypatch.setattr(break_api_pod_order, "lock", lambda *_args: pytest.fail("failed media reached the store write"))
    with pytest.raises(HTTPException) as caught:
        break_api_pod.lock_pod("pod")
    assert caught.value.status_code == 409 and "H123" in caught.value.detail
    assert "H123" in lock_refusal(pod["media"])


def _report(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_import_report_upserts_atomically_and_accepts_feed_aliases(tmp_path) -> None:
    report, store = tmp_path / "qc.csv", tmp_path / "assets.csv"
    _report(report, [{"creative_id": "H123", "duration_seconds": "30", "frame_rate": "25/1", "codec": "xdcamhd422", "qc_state": "approved"}])
    first = import_report(report, store, "MAM QC")
    assert first == {"source": str(report), "store": str(store), "received": 1, "inserted": 1, "updated": 0, "total": 1}
    assert read_assets(store)[0]["frame_rate"] == 25.0
    _report(report, [{"house_number": "H123", "duration_seconds": "29", "video_codec": "xdcamhd422"}])
    second = import_report(report, store, "MAM QC")
    assert second["inserted"] == 0 and second["updated"] == 1 and second["total"] == 1
    assert read_assets(store)[0]["duration_seconds"] == 29.0


def test_import_refuses_unjoinable_empty_or_duplicate_rows(tmp_path) -> None:
    report = tmp_path / "bad.csv"
    _report(report, [{"duration_seconds": "30"}])
    with pytest.raises(MediaReportError, match="House Number"):
        import_report(report, tmp_path / "store.csv")
    _report(report, [{"house_number": "H1", "source": "only provenance"}])
    with pytest.raises(MediaReportError, match="no measured"):
        import_report(report, tmp_path / "store.csv")
    _report(report, [{"house_number": "H1", "duration_seconds": "30"}, {"house_number": "H1", "duration_seconds": "31"}])
    with pytest.raises(MediaReportError, match="more than once"):
        import_report(report, tmp_path / "store.csv")


def test_frontend_prints_failure_and_disables_lock() -> None:
    root = Path(__file__).resolve().parents[1] / "tv-break-dashboard" / "src" / "plan" / "break"
    board = (root / "PodBoard.jsx").read_text(encoding="utf-8")
    verdict = (root / "media" / "MediaVerdict.jsx").read_text(encoding="utf-8")
    assert "mediaBlocked || !onLock" in board
    assert "<MediaLockNotice media={pod.media}" in board
    assert 'role="alert"' in verdict and "blocking_house_numbers" in verdict


def test_round_trip_preserves_columns_the_writer_does_not_own(tmp_path) -> None:
    path = tmp_path / "media_assets.csv"
    write_assets([_asset(probe_build="qc-42")], path)
    assert "probe_build" in path.read_text(encoding="utf-8").splitlines()[0]
    assert read_assets(path)[0]["probe_build"] == "qc-42"
