# Cursor AI – TV-Break Optimisation
# cursor:project tv-break

---

## 1  Project Goals & Guiding Principles
| Goal | What it means for every line of code |
|------|--------------------------------------|
| **Maximise Keshet 12 ad revenue while protecting audience share** | Objective = revenue × viewer-retention; code that ignores churn is rejected. |
| **Hard-separate “expensive training” from “cheap daily optimisation”** | Training may run for hours on GPU; daily query must finish < 5 s, CPU-only, on the frozen posterior. |
| **Design for 10× data growth** | Scope today: 1-month sample → target: ≥ 2 years, minute-level, 4 channels, 45 advertisers. |
| **Observability & reproducibility** | Every run versioned + checksum. One command recreates any result. |

---

## 2  Coding Standards
* **Languages** Python 3.12 (core) · DuckDB SQL (heavy aggs) · tiny Bash/Batch wrappers.
* **Style** `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE` constants.
  * Google-docstrings **and** `#:` type hints, enforced by **mypy --strict**.
  * No magic numbers – keep tunables in **YAML** or `meridian/constants.py`.
* **Dependencies** Pin in `requirements.txt`; new libs require licence + CVE check.
* **Logging** `logging.getLogger(__name__)`; never `print`.
* **Dead code** Forbidden – quarantine in `experimental/` if unsure.

---

## 3  Architecture Preferences
* **Separation of concerns**
  * Each sub-package exposes a single public entry-point (`__all__`).
  * *Training* code must not import *optimiser* code; optimiser only **reads** the frozen posterior.
* **Config-driven** (YAML/ENV → Pydantic models → DI).
* **Pluggability** Each premium/constraint = small strategy class `__call__(Break) -> float`.

---

## 4  Performance Expectations
* **Data transforms** Vectorised Pandas/Polars or DuckDB; no for-loops on >10 k rows.
* **Training** Mixed-precision TensorFlow 2.18 on GPU; checkpoint every 30 min.
* **Daily optimisation** Single-thread, < 2 GB RAM, < 5 s on a modern laptop; deterministic sub-steps memoised (`functools.lru_cache`).
* **Resource hygiene** Close files/DB; `del` + `gc.collect()` on large tensors.

---

## 5  Testing Requirements
| Level | Tooling | Coverage focus |
|-------|---------|----------------|
| **Unit** | `pytest`, `pytest-cov` | Pure funcs, premium calculators, edge cases (zero TVR, >8 min breaks, DST). |
| **Integration** | `pytest` + tiny fixture CSVs | XLSX→CSV transform → Meridian input; train→optimiser round-trip. |
| **E2E smoke** | `make e2e` | 24 h synthetic day; runtime < 90 s; revenue > 0. |
* **Coverage gate** ≥ 90 % overall, 100 % for revenue math.
* **CI** GitHub Actions matrix (Ubuntu 22.04 CPU, optional GPU).

---

## 6  Documentation Standards
* **Code** Docstrings + inline math comments.
* **API** Auto-generate via `pdoc` into `docs/`; every public symbol gets an example.
* **Architecture** Update C4 / Mermaid in `docs/architecture.md` on every structural PR.
* **Changelogs** Human-written `CHANGELOG.md`; CI tags releases (semver).

---

## 7  Contribution Workflow
1. **Branch** `feat/<area>-<ticket>` or `fix/<area>-<ticket>`.
2. **Pre-push hooks** `black` · `isort` · `flake8` · `mypy` · `pytest`.
3. **PR Template** must state goal, approach, trade-offs, test plan, screenshots (if UI).
4. ≥ 1 approval + green CI → merge to `main`.

---

## 8  Cursor AI Prompt Tips
* Always start with `# cursor:project tv-break` then `import this_file`.
* Prefer **small diffs**; ask for a single self-contained function, not 500 LOC.
* Ask for tests first (TDD).
* Provide mini-schema samples (5 rows) for CSV questions.

---

## 9  Golden Rules for Generated Code
1. **Training vs optimisation split** – never blur responsibilities.
2. **Revenue-factor abstraction** – add a new class, *never* hack the pricing loop.
3. **Fail fast** – validate inputs, raise `TVBreakError`.
4. **Zero side-effects** in lib code; CLI handles I/O.
5. **Scalability** – code must work on 2-year, four-channel dataset without edits.

---

## 10  Daily Workflow Inputs & Outputs

### 10.1 Inputs (CSV UTF-8, no BOM)
| File | Purpose |
|------|---------|
| `programme_grid_YYYY-MM-DD.csv` | Full show schedule (all channels, no promo stubs). |
| `keshet_ad_inventory_YYYY-MM-DD.csv` | Keshet 12 ad inventory with constraints (length, position, genre …). |
| `rate_card_premiums.csv` | Static base-rate + premiums table (hour, position, genre, season). |
| `advertiser_rules.csv` | Long-term contracts (per advertiser). |

### 10.2 Outputs
| File | Description |
|------|-------------|
| `optimal_break_plan_keshet12_YYYY-MM-DD.csv` | Break-by-break plan with start, length, spot mix, expected revenue & retention. |
| `summary_YYYY-MM-DD.json` | Totals by hour/daypart (revenue, minutes ads, avg retention). |

---

> **Remember:** if a line of code doesn’t bring us closer to *accurate, scalable* break optimisation—or hides complexity without benchmarking—it’s technical debt.
