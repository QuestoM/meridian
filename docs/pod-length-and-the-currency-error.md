# The pod length, the caps, and what a plan's revenue is worth

An outside reader raised three objections to Kairos in August 2026: that "one ad
break" is undefined and the answer swings by millions; that a fair equal-airtime
comparison makes the engine's plan worse than what the channel actually aired;
and that the optimizer never really optimizes because another break is always
worth more money. Adjudicating them turned up a defect none of the three named,
and this document is the measured record.

Everything here is one month of the operator channel, `data/Spots.csv`, checked
against the shipped plan artifact `output/weekly_break_schedule.csv`.

## 1. What one break is, is in the data

The as-run log carries two columns nothing in this codebase read: `Pos. Block 1`
is a spot's position inside its block, and `Spots Block 1` is that block's size.
Restarting the counter reconstructs the blocks and agrees with the channel's own
reported size on **100 percent of 2,649 pods across all four channels**. There
is no judgement call to make. `kairos/data/as_aired.py` reads it.

Only rows whose `Spot type` is פרסומת carry a block number. Promos, sponsorships
and public-service announcements never do — the channel telling us directly which
airtime it sold.

The claimed sensitivity is real only if you ignore that. Restricted to commercial
rows, the gap threshold barely matters:

| gap threshold | 5s | 15s | 60s | 500s |
| --- | --- | --- | --- | --- |
| pods reconstructed | 786 | 786 | 785 | 780 |

A hundredfold change in the threshold moves the answer under one percent, and all
of them land within 5.2 percent of the 829 ground truth. Reconstructing over
**all** spot rows instead gives 1,703 at 15 seconds — 105 percent above truth —
because it counts unsold airtime as sold. That is where a multi-million swing
comes from, and it is an artefact of the reconstruction, not a property of the
schedule.

## 2. The defect none of the objections named: a currency error

The optimizer plans in two-minute breaks. That number was never measured; it
entered as "a common unit". Measured pod length on the operator channel:

| | seconds |
| --- | --- |
| mean (the planning value) | 190.7 |
| median | 181 |
| lower / upper quartile | 109 / 249 |

Four pods an hour therefore reads as 480 seconds to the engine and is **763 on
air**. The consequence is not a rounding error:

| pod length | seconds at the 4-pod ceiling | 12-min cap | 8-min protected cap |
| --- | --- | --- | --- |
| declared 120s | 480 | clear | clear |
| measured 190.7s | 763 | **breached** | **breached by 59%** |

Run against the real guardrail function on the shipped plan: at 120 seconds,
**zero** hourly-cap violations. At the measured length, **381 of 730 hours breach
(52.2 percent), 132 of them in protected programming** — news and children's
content, where measured pods are longer still (median 238s), not shorter.

The plan is compliant in a unit that does not exist. That is the root defect, and
it also explains the volume: because the seconds cap can never bind at 120s, the
four-pod rule is the only limit, and the engine fills to it in 404 of 553 hours.

## 3. Volume and placement must be measured separately

At its own volume the plan schedules 2.70x the pods and 1.70x the seconds the
channel aired, and delivers **−33.9 percent** rating per commercial second. That
reads as a damning verdict on the engine. It is not one.

Hand the optimizer the channel's own day-by-day break budget — 829 against 829 —
and it delivers **+27.0 percent** rating per commercial second. The placement is
genuinely better than the human schedule; what fails is the volume assumption.

Any plan-versus-actual comparison that does not hold airtime fixed is measuring
the volume assumption, not the optimizer. This is the trap the outside reader
fell into in one direction and the engine's own revenue figure falls into in the
other.

## 4. The optimizer does allocate; the retention model does not bind

Marginal net revenue is positive throughout the cap for **960 of 978** eligible
segments, exactly as reported. The unconstrained turning point sits at a median
of **11** pods per segment, far above any legal cap of 4, so viewer-retention cost
never bites inside the legal envelope. That much is a real finding.

The conclusion drawn from it is not. The plan does **not** fill to the maximum:
33.2 percent of eligible segments end at their cap, 66.8 percent end below, 77
get zero, and the plan uses **60.4 percent** of the per-segment ceiling. The
binding rules are the hour and day guardrails, so the real decision is which
segments get scarce pods — worth, at an identical break budget, **+21.35M** over
broadcast order and **+7.18M** over spreading evenly.

## 5. Why one number and not a daypart model

Pod length looks strongly daypart-shaped: 42 seconds overnight, 371 at ten in the
evening, with hour explaining 29.8 percent of in-sample variance and hour crossed
with weekday 41.0 percent. Held out — trained on 21 days, tested on 9 — the best
shrunk hour model beats a single global constant by **4.1 percent of mean absolute
error**, 76.2 against 79.5 seconds. The cells behind it are thin (16 pods in the
one o'clock hour). This codebase does not ship a model that cannot beat a constant
out of sample, so the shipped value is one measured number and the seam takes a
richer estimate when a larger sample earns one.

## 6. What activation does, measured

`measured_pod_length_activation` is **off by default**; off, the assumptions object
is returned unchanged and every existing figure is byte-identical (verified: the
off path reproduces the shipped artifact's 2,239 pods, 268,680 seconds and 34.71M
exactly). Activated, on the same month:

| | off (120s) | on (190.7s) |
| --- | --- | --- |
| pods planned | 2,239 | 1,500 (−33.0%) |
| commercial airtime | 74.6 h | 79.5 h (+6.5%) |
| revenue | 34.71M | 43.47M (**+25.2%**) |
| hourly-cap violations | 0 | 0 |
| versus what the channel aired | 2.70x pods | 1.81x pods |

Revenue rises by more than airtime does — 17.6 percent of it is revenue per second,
not extra seconds — because a constraint that finally binds forces the optimizer to
choose, and choosing is the thing it is good at. The honest fix is also the more
profitable one, which is rare enough to state carefully: it is a consequence of the
caps beginning to work, not of pricing anything higher.

**What activation does not fix.** The plan still schedules 1.81x the airtime the
channel actually sells. The revenue figure is the value of that volume and is not
an increment on today's business. `kairos/model/plan_against_aired.py` now puts
that comparison in the same payload as the revenue rather than a screen away.

## 7. Open, and not ours to close

`data/regulatory_guardrails.json` states a 12-minute hourly cap;
`docs/campaign-rate-card-research.md` states 10. Both cannot be right, and the
difference decides how hard the cap bites at real pod length. This needs the
regulator's own text, not a measurement.
