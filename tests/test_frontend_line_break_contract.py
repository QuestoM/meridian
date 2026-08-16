from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = ROOT / "tv-break-dashboard" / "src"
UI_SOURCE = tuple(
    path
    for pattern in ("*.jsx", "*.tsx", "*.js", "*.ts")
    for path in FRONTEND_SRC.rglob(pattern)
) + (ROOT / "tv-break-dashboard" / "index.html",)
CSS_SOURCE = tuple(FRONTEND_SRC.rglob("*.css"))


def test_ui_copy_does_not_art_direct_lines_with_break_tags_or_hard_spaces():
    """Text wraps from layout and type metrics, never authored line breaks."""
    offenders = []
    break_tag = re.compile(r"<br\b", re.IGNORECASE)
    hard_space = re.compile(
        "(?:&nbsp;|&#160;|&#x0*a0;|\\u00a0|\\\\u00a0)",
        re.IGNORECASE,
    )

    for path in UI_SOURCE:
        source = path.read_text(encoding="utf-8")
        if break_tag.search(source) or hard_space.search(source):
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == [], (
        "Remove art-directed line breaks and hard spaces; let the container, "
        f"typography, and responsive layout wrap the text: {offenders}"
    )


def test_ui_copy_is_not_capped_to_an_authored_number_of_lines():
    """A component may constrain space, but it must not prescribe line count."""
    offenders = []
    fixed_line_count = re.compile(r"(?:^|[^\w])(?:-webkit-)?line-clamp\s*:", re.MULTILINE)

    for path in CSS_SOURCE:
        if fixed_line_count.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == [], (
        "Remove fixed line clamps; text should wrap from its container and type "
        f"metrics: {offenders}"
    )


def test_page_intro_copy_can_wrap_in_its_container():
    """Page-level explanatory prose must not be silently forced onto one line."""
    cases = (
        (
            FRONTEND_SRC / "sources" / "studio-ledger-sources.css",
            ".sources-page > .page-header p",
        ),
        (
            FRONTEND_SRC / "plan" / "day" / "master-control-broadcast.css",
            ".broadcast-library > .page-header p",
        ),
        (
            FRONTEND_SRC / "plan" / "day" / "master-control-broadcast.css",
            ".broadcast-decisions > .page-header p",
        ),
        (
            FRONTEND_SRC / "plan" / "day" / "master-control-broadcast.css",
            ".broadcast-pods .pod-page-head p",
        ),
    )

    offenders = []
    for path, selector in cases:
        source = path.read_text(encoding="utf-8")
        match = re.search(re.escape(selector) + r"[^{}]*\{([^{}]*)\}", source)
        assert match is not None, f"Missing narrative selector: {selector}"
        block = match.group(1)
        if re.search(r"white-space\s*:\s*nowrap", block) or re.search(
            r"text-overflow\s*:\s*ellipsis", block
        ):
            offenders.append(f"{path.relative_to(ROOT)}: {selector}")

    assert offenders == [], (
        "Page intro copy must wrap naturally instead of being painted as a "
        f"single ellipsized line: {offenders}"
    )


def test_narrative_rows_do_not_recreate_break_tags_with_layout_rules():
    """Sibling text fragments stay in one wrapping flow at every width."""
    cases = (
        (
            FRONTEND_SRC / "plan" / "day" / "master-control-broadcast.css",
            ".broadcast-decisions .oc-row-meta",
            (r"flex-direction\s*:\s*column",),
        ),
        (
            FRONTEND_SRC / "plan" / "week" / "plan-week-recommendations.css",
            ".plan-recommendation-row small",
            (r"white-space\s*:\s*nowrap", r"text-overflow\s*:\s*ellipsis"),
        ),
        (
            FRONTEND_SRC / "plan" / "week" / "plan-week-instruments.css",
            ".plan-week .plan-goal-basis",
            (r"white-space\s*:\s*nowrap", r"text-overflow\s*:\s*ellipsis"),
        ),
        (
            FRONTEND_SRC / "plan" / "week" / "plan-week-instruments.css",
            ".plan-week .plan-goal-rule",
            (
                r"flex-basis\s*:\s*100%",
                r"white-space\s*:\s*nowrap",
                r"text-overflow\s*:\s*ellipsis",
            ),
        ),
    )

    offenders = []
    for path, selector, banned in cases:
        source = path.read_text(encoding="utf-8")
        match = re.search(re.escape(selector) + r"[^{}]*\{([^{}]*)\}", source)
        assert match is not None, f"Missing narrative selector: {selector}"
        block = match.group(1)
        if any(re.search(pattern, block) for pattern in banned):
            offenders.append(f"{path.relative_to(ROOT)}: {selector}")

    assert offenders == [], (
        "Narrative fragments must share a naturally wrapping flow rather than "
        f"recreate authored line breaks in CSS: {offenders}"
    )


def test_long_narrative_fragments_can_shrink_and_wrap_inside_their_real_container():
    """The wrapping flow must also let a long flex/grid child become narrower."""
    checks = (
        (
            FRONTEND_SRC / "plan" / "day" / "master-control-broadcast.css",
            ".broadcast-decisions .oc-row-meta > span",
            (r"min-inline-size\s*:\s*0", r"overflow-wrap\s*:\s*anywhere"),
        ),
        (
            FRONTEND_SRC / "plan" / "week" / "plan-week-recommendations.css",
            ".plan-recommendation-row small",
            (r"white-space\s*:\s*normal", r"overflow-wrap\s*:\s*anywhere"),
        ),
        (
            FRONTEND_SRC / "plan" / "week" / "plan-week-instruments.css",
            ".plan-week .plan-goal-rule",
            (r"white-space\s*:\s*normal", r"overflow-wrap\s*:\s*anywhere"),
        ),
    )

    for path, selector, required in checks:
        source = path.read_text(encoding="utf-8")
        match = re.search(re.escape(selector) + r"[^{}]*\{([^{}]*)\}", source)
        assert match is not None, f"Missing narrative selector: {selector}"
        for pattern in required:
            assert re.search(pattern, match.group(1)), f"{path.relative_to(ROOT)}: {selector} lacks {pattern}"
