"""P4: the form fields on the Clients destination keep their direction and name.

The measured defect. Every editable field in this destination's four MUI forms
passed its html attributes through ``inputProps``, which the installed MUI major
no longer forwards to the input. Measured in a browser on the shipped bundle,
with the advertiser record drawer open: five inputs, ``dir`` null on all five,
``aria-label`` null on all five, ``min`` and ``step`` null on the three numeric
ones, five DOM nodes carrying a stray ``inputprops`` attribute, and one React
console error. A number field with no ``dir`` renders its caret and its minus
sign on the wrong side inside an RTL page, and a field with no accessible name
is unreachable by name to a screen reader.

After the fix, measured the same way: ``dir="ltr"`` on all three numeric inputs,
``min`` and ``step`` present, an ``aria-label`` on all five, zero stray
attributes and zero console errors.

This file guards the invariant behind that measurement rather than the copy: the
deprecated prop never comes back, and every field that must read left to right
inside a right to left page still says so. The MUI major is asserted first, so
if a future downgrade makes ``inputProps`` correct again this file says which
version it measured rather than failing silently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tv-break-dashboard"
CLIENTS = APP / "src" / "clients"

# The four forms this destination edits records through.
FORMS = [
    "AddAdvertiserForm.jsx",
    "AdvertiserConditions.jsx",
    "AdvertiserDetailDrawer.jsx",
    "AgencyDetailDrawer.jsx",
]

# Fields that hold a number, an id or an email inside an RTL page. Each one is
# named by the label the operator reads, so a failure names the field.
LTR_FIELDS = [
    "Default premium multiplier",
    "Advertiser ID",
    "Behind-pace pacing strength",
    "Over-delivery pacing restraint",
    "Coefficient value",
    "Surcharge discount percent",
    "Placement preference percent",
]


def _sources() -> dict[str, str]:
    missing = [name for name in FORMS if not (CLIENTS / name).is_file()]
    if missing:
        pytest.skip(f"these forms are not in this tree: {', '.join(missing)}")
    return {name: (CLIENTS / name).read_text(encoding="utf-8") for name in FORMS}


def test_the_installed_mui_major_is_the_one_that_drops_the_old_prop() -> None:
    """The premise. Below major 6 the old prop still reached the input."""
    manifest = APP / "node_modules" / "@mui" / "material" / "package.json"
    if not manifest.is_file():
        pytest.skip("the dashboard's node_modules is not installed")
    version = json.loads(manifest.read_text(encoding="utf-8"))["version"]
    assert int(version.split(".")[0]) >= 6, version


def test_no_form_passes_html_attributes_through_the_dropped_prop() -> None:
    """The defect itself, which lands the attributes on nothing."""
    for name, source in _sources().items():
        assert "inputProps=" not in source, f"{name} still passes inputProps"


def test_every_field_that_must_read_left_to_right_still_says_so() -> None:
    """The RTL half: a number or an id inside a Hebrew page is dir ltr."""
    joined = "\n".join(_sources().values())
    for label in LTR_FIELDS:
        # The direction and the accessible name travel together in one object,
        # so the assertion is that this label's own object still carries ltr.
        window = [
            block for block in re.findall(r"\{[^{}]*'aria-label'[^{}]*\}", joined)
            if label in block
        ]
        assert window, f"no field object carries the label {label}"
        assert all("dir: 'ltr'" in block for block in window), label


def test_every_field_object_now_travels_inside_the_slot_it_belongs_to() -> None:
    """The fix: the html input slot for a text field, the input slot for a switch."""
    sources = _sources()
    assert "slotProps={{ input: {" in sources["AddAdvertiserForm.jsx"]
    assert "slotProps={{ input: {" in sources["AdvertiserDetailDrawer.jsx"]
    for name, source in sources.items():
        if name == "AdvertiserConditions.jsx":
            assert "htmlInput: isSurchargeDiscount" in source
        assert "htmlInput" in source, f"{name} has no html input slot"
