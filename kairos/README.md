# Kairos engine

Kairos is the backend that helps an Israeli TV channel decide, each day, where to
place commercial breaks and how long they run, to maximize ad revenue times
viewer retention (TVR rating points) under regulatory and policy guardrails. It
resolves the daily tension between marketing, which wants more and longer breaks,
and program managers, who want fewer and shorter ones. It is built toward Google
Meridian (Bayesian MMM): the KPI is TVR retention, and the media channels are
program type by break position by break length.

The reference data covers four channels for November 2024: Keshet 12, Reshet 13,
Kan 11, and Now 14.

## Principles

- No fabrication. Missing or unknown data is reported or shown as an honest zero,
  empty, or null. It is never replaced with a plausible-looking number. A
  programme with no rating contributes a segment the optimizer simply will not
  load, rather than an invented value.
- Transparent decisions. The optimizer is a greedy marginal-value allocator, not
  an opaque solver, so every break it places carries the gain that justified it.
- Declared assumptions. Retention-side defaults are labelled as assumptions until
  the Meridian impact model is trained, and they travel with each segment.

## Data flow

```
loaders -> contracts -> classifier -> transform -> objective + guardrails
        -> optimizer -> pricing -> export -> service -> kairos_api
```

## Modules

- `data/loaders.py` reads the real Spots, Programmes, and Dayparts xlsx and the
  Hebrew daily-input csv into canonical frames.
- `data/contracts.py` validates loaded frames against the documented schema and
  returns a report rather than raising, so data issues are visible and honest.
- `data/classifier.py` plus `config/program_categories.yaml` assign each
  programme title a genre with weighted, Hebrew-aware scoring and an honest
  "Other" when confidence is low.
- `data/transform.py` turns the programme grid into optimizer `ProgramSegment`s,
  carrying the real TVR and a positional pricing class.
- `optimize/objective.py` holds the revenue and retention primitives (CPP break
  revenue, fixed revenue, predicted retention, weighted objective).
- `optimize/guardrails.py` holds the regulatory and policy constraints as pure
  check functions returning typed violations.
- `optimize/optimizer.py` is the greedy allocator: it adds the single
  highest-objective-gain break each step, only if the schedule stays compliant.
- `optimize/pricing.py` is a typed view over `config/optimization_weights.yaml`
  plus the declared `OptimizerAssumptions`.
- `optimize/agreements.py` models advertiser FIX and sponsorship commitments and
  must-air constraints, with honest empty handling when the agreement files are
  stubs.
- `model/` is the env-gated seam to a trained Meridian posterior. It imports
  cleanly without Meridian or TensorFlow and falls back to the declared
  assumption coefficient when no fitted model is present. `model/prepare.py`
  shapes the real Spots, Programmes, and Dayparts into a Meridian `InputData`
  keyed by the engine channels (program_type by break_position by break_length);
  `scripts/train_impact_model.py` fits the posterior; `service.py` loads it and
  threads the per-channel coefficient into each segment.
- `observability/` records each run with input checksums, the engine version, the
  guardrails and assumptions used, and the headline metrics, for audit and
  reproducibility.
- `export/schedule.py` runs the real engine across every channel-day and writes
  `output/weekly_break_schedule.csv`, which the dashboard reads.
- `service.py` is the single seam the API calls. It runs load to classify to
  price to optimize and returns plain dicts.

## Running

Fast unit gate (no network, no xlsx):

```
cd kairos-build
PYTHONUTF8=1 python -m pytest tests/ -q \
  --ignore=tests/test_loaders.py \
  --ignore=tests/test_transform_integration.py \
  --ignore=tests/test_api_engine.py
```

Regenerate the real weekly schedule csv from the reference data:

```
PYTHONUTF8=1 python scripts/export_schedule.py
```

This is an offline batch (a few minutes over the full month). The dashboard reads
the committed csv instantly.

## The retention impact model

Each break carries a per-channel retention impact coefficient (how much audience
that break sheds). The optimizer resolves it in this order, most authoritative
first, all logged so the source is never hidden:

1. **Measured coefficients** (`models/tv_break_coefficients.json`). The real,
   detrended, pooled per-break effect computed from the minute-level audience
   curve. This needs no TensorFlow or Meridian, so it runs on any Python. It is
   the recommended source. `service.py` reports `impact_source="measured"`.
2. **Trained Meridian posterior** (`models/tv_break_posterior.pkl`). A fitted
   Bayesian MMM, mapped into retention deltas. `impact_source="trained"`.
3. **Declared assumption**, the honest fallback when neither file is present.

### Measuring the coefficients (fast, no heavy stack)

```
cd kairos-build
PYTHONUTF8=1 python scripts/compute_measured_coefficients.py
```

For each real break this measures the mean audience just before it and just after
content resumes, divides out the typical audience trajectory at that broadcast
minute (so the prime-time ramp does not masquerade as a break benefit), pools
thin channel cells toward the global mean, and writes the JSON. See
`kairos/model/measure.py`.

### Training the Meridian posterior (offline, Python 3.11/3.12 + TensorFlow)

The desktop default Python is 3.13 and cannot train; `model/train.py` raises a
clear error when the stack is absent. To fit it:

```
cd kairos-build
py -3.11 -m venv .venv311
.venv311/Scripts/python -m pip install -e . pyyaml openpyxl
PYTHONUTF8=1 .venv311/Scripts/python scripts/train_impact_model.py
```

This builds the InputData from the real reference data, samples the posterior,
and writes `models/tv_break_posterior.pkl`.
