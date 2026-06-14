"""Kairos data pipeline: loaders, classifier, transforms, and data contracts."""

from kairos.data.classifier import Classification, ProgramClassifier
from kairos.data.contracts import (
    ValidationReport,
    Violation,
    validate_daily_input,
    validate_dayparts,
    validate_programmes,
    validate_spots,
)
from kairos.data.transform import build_segments_from_programmes

__all__ = [
    "Classification",
    "ProgramClassifier",
    "ValidationReport",
    "Violation",
    "build_segments_from_programmes",
    "validate_daily_input",
    "validate_dayparts",
    "validate_programmes",
    "validate_spots",
]
