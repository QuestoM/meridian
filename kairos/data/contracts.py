"""Schema-validation (data contracts) for Kairos loaded DataFrames.

These functions validate the SHAPE the loaders in ``kairos.data.loaders``
actually produce, not the raw source files. Each validator returns a typed
``ValidationReport`` instead of raising, so callers can decide how strict to
be. The guiding rule is honesty: missing or NaN data is reported as a warning
or error, never silently replaced with a plausible-looking number.

Reference for the documented columns/types is ``data_schemas.yaml`` at the
repo root. The validators below align to the post-load column names:

  - Programmes: Title, Channel, Date, Start time, End time, Duration, TVR,
    start_dt, end_dt
  - Spots: Campaign, Channel, Date, Start time, Duration, ... , TVR, air_dt
  - Dayparts (melted long): date, timeband, channel, tvr
  - Daily input (Hebrew renamed): date, advertiser, campaign, program,
    duration_sec, position_in_break, planned_tvr, price, status, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pandas as pd

try:  # The known Israeli channel set, if importable.
    from kairos.data.loaders import CHANNELS as _LOADER_CHANNELS

    KNOWN_CHANNELS: tuple[str, ...] = tuple(_LOADER_CHANNELS)
except Exception:  # pragma: no cover - loaders should always import.
    KNOWN_CHANNELS = ()


SEVERITIES = ("error", "warning")


@dataclass(frozen=True)
class Violation:
    """A single data-contract finding.

    Attributes:
      field: The column or frame-level scope the finding refers to.
      code: A short stable machine code, e.g. "missing_column".
      detail: A human-readable explanation of the finding.
      severity: One of "error" or "warning".
    """

    field: str
    code: str
    detail: str
    severity: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"severity must be one of {SEVERITIES}, got {self.severity!r}"
            )

    def __str__(self) -> str:
        return f"[{self.severity}] {self.field}: {self.code} - {self.detail}"


@dataclass
class ValidationReport:
    """The outcome of validating one DataFrame against a contract.

    ``is_valid`` is True when there are no error-severity violations. Warnings
    do not invalidate the report; they flag honest data-quality concerns such
    as NaN ratings that downstream code should handle explicitly.
    """

    dataset: str
    violations: list[Violation] = field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add(self, field: str, code: str, detail: str, severity: str) -> None:
        self.violations.append(Violation(field, code, detail, severity))

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        header = (
            f"ValidationReport({self.dataset}): {status} "
            f"[{len(self.errors)} error(s), {len(self.warnings)} warning(s)]"
        )
        if not self.violations:
            return header + "\n  (no findings)"
        lines = [header]
        for violation in self.violations:
            lines.append("  " + str(violation))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared checks
# ---------------------------------------------------------------------------


def _check_is_frame(df: object, report: ValidationReport) -> bool:
    """Return True if df is a non-None DataFrame; otherwise record an error."""
    if df is None:
        report.add("<frame>", "missing_frame", "DataFrame is None", "error")
        return False
    if not isinstance(df, pd.DataFrame):
        report.add(
            "<frame>",
            "not_a_frame",
            f"expected DataFrame, got {type(df).__name__}",
            "error",
        )
        return False
    return True


def _require_columns(
    df: pd.DataFrame, columns: Sequence[str], report: ValidationReport
) -> list[str]:
    """Record an error per missing column. Return the present columns."""
    present = []
    for column in columns:
        if column in df.columns:
            present.append(column)
        else:
            report.add(
                column,
                "missing_column",
                "required column is absent from the frame",
                "error",
            )
    return present


def _check_numeric_coercible(
    df: pd.DataFrame, column: str, report: ValidationReport
) -> pd.Series | None:
    """Return a numeric Series for column, recording how coercion went.

    Values that cannot be coerced become NaN. We never invent a replacement;
    we only count how many entries were lost so the caller can react.
    """
    if column not in df.columns:
        return None
    original = df[column]
    coerced = pd.to_numeric(original, errors="coerce")
    # Entries that were non-null before but null after are coercion failures.
    lost = int((original.notna() & coerced.isna()).sum())
    if lost:
        report.add(
            column,
            "non_numeric_values",
            f"{lost} value(s) could not be coerced to numeric",
            "error",
        )
    return coerced


def _warn_on_nan(
    series: pd.Series, column: str, report: ValidationReport
) -> None:
    """Record a warning (never an error) for NaN entries in a numeric column."""
    nan_count = int(series.isna().sum())
    if nan_count:
        report.add(
            column,
            "nan_values",
            f"{nan_count} NaN value(s) present; not imputed",
            "warning",
        )


def _check_non_negative(
    series: pd.Series, column: str, report: ValidationReport
) -> None:
    """Record an error for any value strictly below zero (NaN ignored)."""
    negative = int((series < 0).sum())
    if negative:
        report.add(
            column,
            "negative_values",
            f"{negative} value(s) below zero",
            "error",
        )


def _check_strictly_positive(
    series: pd.Series, column: str, report: ValidationReport
) -> None:
    """Record an error for any value at or below zero (NaN ignored)."""
    non_positive = int((series <= 0).sum())
    if non_positive:
        report.add(
            column,
            "non_positive_values",
            f"{non_positive} value(s) at or below zero",
            "error",
        )


def _check_datetime(
    df: pd.DataFrame, column: str, report: ValidationReport
) -> None:
    """Record an error if the column is present but not datetime-typed."""
    if column not in df.columns:
        return
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        report.add(
            column,
            "not_datetime",
            "expected a datetime64 dtype",
            "error",
        )


def _check_channels(
    values: Iterable[object], column: str, report: ValidationReport
) -> None:
    """Warn for any channel value outside the known Israeli set.

    Treated as a warning, not an error: new channels can legitimately appear
    in the data before the known set is updated. NaN channels are flagged too.
    """
    if not KNOWN_CHANNELS:
        return
    seen = pd.Series(list(values))
    nan_count = int(seen.isna().sum())
    if nan_count:
        report.add(
            column,
            "nan_channel",
            f"{nan_count} row(s) have a missing channel",
            "warning",
        )
    unknown = sorted(
        {str(v) for v in seen.dropna().unique() if str(v) not in KNOWN_CHANNELS}
    )
    if unknown:
        report.add(
            column,
            "unknown_channel",
            f"channel(s) not in known set: {unknown}",
            "warning",
        )


# ---------------------------------------------------------------------------
# Public validators
# ---------------------------------------------------------------------------


def validate_programmes(df: pd.DataFrame) -> ValidationReport:
    """Validate the frame produced by ``loaders.load_programmes``."""
    report = ValidationReport("programmes")
    if not _check_is_frame(df, report):
        return report

    required = ["Title", "Channel", "Duration", "TVR", "start_dt", "end_dt"]
    _require_columns(df, required, report)

    _check_datetime(df, "start_dt", report)
    _check_datetime(df, "end_dt", report)

    if "start_dt" in df.columns and "end_dt" in df.columns:
        both = df.dropna(subset=["start_dt", "end_dt"])
        backwards = int((both["end_dt"] < both["start_dt"]).sum())
        if backwards:
            report.add(
                "end_dt",
                "end_before_start",
                f"{backwards} programme(s) end before they start",
                "error",
            )

    duration = _check_numeric_coercible(df, "Duration", report)
    if duration is not None:
        _warn_on_nan(duration, "Duration", report)
        _check_strictly_positive(duration, "Duration", report)

    tvr = _check_numeric_coercible(df, "TVR", report)
    if tvr is not None:
        _warn_on_nan(tvr, "TVR", report)
        _check_non_negative(tvr, "TVR", report)

    if "Channel" in df.columns:
        _check_channels(df["Channel"], "Channel", report)

    return report


def validate_spots(df: pd.DataFrame) -> ValidationReport:
    """Validate the frame produced by ``loaders.load_spots``."""
    report = ValidationReport("spots")
    if not _check_is_frame(df, report):
        return report

    required = ["Campaign", "Channel", "Duration", "TVR", "air_dt"]
    _require_columns(df, required, report)

    _check_datetime(df, "air_dt", report)

    duration = _check_numeric_coercible(df, "Duration", report)
    if duration is not None:
        _warn_on_nan(duration, "Duration", report)
        _check_strictly_positive(duration, "Duration", report)

    tvr = _check_numeric_coercible(df, "TVR", report)
    if tvr is not None:
        _warn_on_nan(tvr, "TVR", report)
        _check_non_negative(tvr, "TVR", report)

    if "Channel" in df.columns:
        _check_channels(df["Channel"], "Channel", report)

    return report


def validate_dayparts(df: pd.DataFrame) -> ValidationReport:
    """Validate the long-form frame produced by ``loaders.load_dayparts``."""
    report = ValidationReport("dayparts")
    if not _check_is_frame(df, report):
        return report

    required = ["date", "timeband", "channel", "tvr"]
    _require_columns(df, required, report)

    _check_datetime(df, "date", report)

    tvr = _check_numeric_coercible(df, "tvr", report)
    if tvr is not None:
        _warn_on_nan(tvr, "tvr", report)
        _check_non_negative(tvr, "tvr", report)

    if "channel" in df.columns:
        _check_channels(df["channel"], "channel", report)

    return report


def validate_daily_input(df: pd.DataFrame) -> ValidationReport:
    """Validate the frame produced by ``loaders.load_daily_input``.

    The ``price`` and ``status`` columns are optimizer outputs and are
    intentionally empty in the input, so their NaNs are not flagged here.
    """
    report = ValidationReport("daily_input")
    if not _check_is_frame(df, report):
        return report

    required = ["date", "advertiser", "campaign", "program", "duration_sec"]
    _require_columns(df, required, report)

    _check_datetime(df, "date", report)

    duration = _check_numeric_coercible(df, "duration_sec", report)
    if duration is not None:
        _warn_on_nan(duration, "duration_sec", report)
        _check_strictly_positive(duration, "duration_sec", report)

    planned = _check_numeric_coercible(df, "planned_tvr", report)
    if planned is not None:
        _warn_on_nan(planned, "planned_tvr", report)
        _check_non_negative(planned, "planned_tvr", report)

    # position_in_break is a nullable optimizer-target field; it is legitimately
    # empty or zero in the raw input, so only negatives are flagged.
    position = _check_numeric_coercible(df, "position_in_break", report)
    if position is not None:
        _check_non_negative(position, "position_in_break", report)

    return report
