# The goal-based order, and the seam it reaches the engine through

Written 2026-08-09 from measurement on this tree. This is the contract both
sides of the seam publish, in the sense `docs/constraint-predicate-contract.md`
is a contract: the engine side and the commercial side each name what they
promise, and neither is free to change it quietly.

The trade's own words outrank this document, and they say:

> The stated direction is that the channel takes over placement: the agency sends
> a GRP or target-audience goal instead of a spot list, and the channel is
> accountable for delivering it under all its constraints. This is the product's
> real thesis, and the goal-based order is therefore not a secondary mode. It is
> the destination.

---

## 1. What was already true, measured before the first edit

| What | Measured on this tree |
|---|---|
| `data/campaigns.csv` campaign rows | 52 |
| Campaign rows carrying `rating_goal_points` and `rating_goal_audience` | 51 |
| Of those, rows marked `is_demo` | **51. All of them.** |
| Real, non-demo goal-based orders | **0** |
| `data/campaign_delivery.csv` day rows | 368, across 51 campaigns |
| Of those, `air_state` unknown / aired / scheduled | 306 / 44 / 18 |
| `data/campaign_flights.csv` rows | 0. Header only, owner decision 4 |
| Times the optimizer read any goal field | **0** |

The goal was already modelled, stored, served, paced against and settled
against. It had never once reached the engine that places breaks. That last row
is the whole piece.

Two consequences follow and both are load-bearing.

**The seam ships inert.** Every goal on disk is a demo row, and a seeded row is
not a booking, so `load_goal_orders()` returns an empty list and no shipped plan
can move. The identity is arithmetic, not a flag.

**Delivery is mostly unknown.** 306 of 368 ledger days carry no per-spot source.
So what a campaign has delivered is a FLOOR almost everywhere, and what it has
left to place is a CEILING. Both travel with their basis wherever they are shown.

---

## 2. What a goal-based order is

An order that states an outcome and books no lines. It is **complete** that way.
That is the point of it: the agency states the goal and the channel owns the
placement.

The product publishes which of three kinds an order is, on every campaign record,
rather than leaving a surface to infer it from a blank:

| kind | when | complete |
|---|---|---|
| `goal_based` | carries a positive `rating_goal_points`, books no lines | yes |
| `spot_list` | any flight books `spots` or `seconds` | yes |
| `not_an_order_yet` | neither | no, with the path forward |

A flight stating `grp` or `impressions` is another way of writing the same
outcome, so it does not turn a goal-based order into a spot-list one. Only
`spots` and `seconds` are booked lines.

This matters because of a trade fact that binds harder here than anywhere:

> Agencies over-order prime deliberately. They ask for more prime minutes than
> they need because they know they will not get them, so the remainder lands
> where they actually wanted it. An order quantity is a negotiating position, not
> a demand forecast.

A goal-based order has no negotiating position in it. The number a trader types
means what it says, for the first time, and the product must plan to deliver it
rather than to discount it. That sentence is published in both languages on every
goal-based order record, so no surface treats the figure as an upper bound.

---

## 3. The seam, in full

Engine side: `kairos/optimize/goal_seam.py`. It is the seam and only the seam. It
holds no placement logic, no objective and no greedy step.

Commercial side: `kairos_api/campaigns_goal_order.py` and
`kairos_api/campaigns_goal_words.py`, beside the store that persists the goal.

### 3.1 What the engine side promises

```python
GoalOrder(campaign_id, channel, audience, goal_points,
          starts_on, ends_on, status, is_demo, priority, pacing_mode)
DeliveredPoints(campaign_id, points_counted, days_counted, days_unknown, days_total)
GoalFeasibility(campaign_id, state, basis, required_per_day,
                supply_per_day, share_of_supply, days_left, unmet_points)

load_goal_orders(path=None, *, include_demo=False) -> list[GoalOrder]
load_delivered_points(path=None) -> dict[str, DeliveredPoints]

days_left(order, today) -> int | None
remaining_days(order, today) -> list[str]
unmet_points(order, delivered, today) -> tuple[float | None, basis]
day_pressure(supply, orders, day, channel, today, delivered_of) -> float
goal_feasibility(order, delivered, today, supply_per_day) -> GoalFeasibility

build_goal_weights(segments, orders, today, *, delivered_of, k, u_max, u_min)
    -> dict[segment_id, float]
fold_into_demand_weights(weights, segments, today, *, orders, delivered_of, ...)
    -> dict[segment_id, float]
goal_adjusted_net(net_of, segments, orders, today, *, delivered_of, shadow)
    -> Callable[[ProgramSegment, int], float]

seam_state(orders) -> dict
```

Guarantees, each of which has a test:

1. `today is None`, no orders, or every order demo, gives the identity. For
   `goal_adjusted_net` the identity is the SAME OBJECT, not merely equal numbers.
2. An audience with no panel behind it contributes nothing. An unknown is never
   spent as though it were zero.
3. An order whose `channel` is empty or is not the segment's channel steers
   nothing. That is the competitor boundary inside this seam; the channel on a
   campaign row is written from settings through `channel_scope.operator_channel`.
4. A day whose in-scope segments all carry the same expected rating is an exact
   identity. There is no rating-efficiency difference for a points goal to prefer.
5. Nothing here reads a clock or calls random. `today` is the caller's.
6. Nothing here changes reported revenue. Both halves touch ranking or the
   objective scalar only; revenue is built in `_build_result` from
   `_segment_revenue` and does not pass through the seam.

### 3.2 What the commercial side promises

`order_block(commitment, flights)` publishes the order kind and its completeness
on every campaign record, with the "no spot list is not missing data" sentence in
both languages.

`goal_order_read(order, ...)` answers "what will this goal take, and on what
basis". It is a SUPPLY verdict, never a promise:

```
required_per_day = unmet_points / days_left
supply_per_day   = the channel's own expected rating on those days, from the plan
share_of_supply  = required_per_day / supply_per_day
```

`fits` below 0.5, `tight` from 0.5 to 1.0, `exceeds_supply` above 1.0, `unknown`
when any input cannot be derived. Every refusal is READ from
`kairos_api.pacing_alerts_api_words.reason()`, which already publishes the
product's own sentence for an unmeasurable audience, a flight with no dates and a
gap in the elapsed days. Nothing paraphrases them.

Why a supply verdict and not a delivery forecast: the weekly plan holds break
counts and has no per-campaign line, so the product cannot derive how many points
THIS order will receive. It says so in the payload rather than letting a reader
assume otherwise.

---

## 4. Why a goal moves placement, and the two halves of the seam

Revenue is `cpp * rating_points`. A rating-point goal is `rating_points` alone.
Those are not the same ordering. A large audience in a cheap daypart is efficient
for a points goal and inefficient for revenue. That divergence is the entire
mechanism, and there is nothing else in the seam.

### 4.1 The ranking half, and the measured reason it is not enough

`build_goal_weights` produces the per-segment lean:

```
supply    = sum of baseline_tvr over the day's in-scope segments
mean_tvr  = supply / count
pressure  = clamp(sum over orders of (unmet / days_left) / supply, 0, 1)
weight(s) = clamp(1 + K * pressure * (tvr(s) / mean_tvr - 1), U_MIN, U_MAX)
```

`K = 1.0`, `U_MIN = 0.5`, `U_MAX = 2.0`, the same shape as the pacing knobs. The
`(ratio - 1)` term is what makes it differential; a uniform multiplier on a day
changes no ranking and would be a silent no-op.

**This half is measurably inert on the shipped path, and not because of anything
in this piece.** Measured on the operator channel, 2024-11-04, 88 segments, with
a goal weight map varying from 0.77 to 2.00 across the day:

| daily ad-seconds cap | refine | segments moved | goal points |
|---|---|---|---|
| 9600 (the default) | off | 0 | +0.000% |
| 9600 | on | 0 | +0.000% |
| 6000 | off | 0 | +0.000% |
| 6000 | on | 0 | +0.000% |
| 4800 | off | **2** | **+0.229%** |
| 4800 | on | 0 | +0.000% |
| 2400 | off | 0 | +0.000% |
| 2400 | on | 0 | +0.000% |

Two findings, and the second is the important one.

**A demand weight only bites when a global cap binds.** Below the cap the greedy
takes every positive-gain break whatever order it ranks them in, so the final
schedule is the same set.

**Where it does bite, the refiner takes it back out.** `refine=True` is the
default on every shipped path. The F1 refiner and the exact DP tier both climb
`_group_objective_contribution` (or the net in net mode), and neither reads
`demand_weights`. So any bias the greedy ranking took on is optimised straight
back out wherever the refiner can improve the true objective.

That applies to the whole class, not just to the goal: **advertiser demand,
inventory awareness and delivery pacing are subject to the same erasure.** It is
reported here because it was measured here, and it is not this piece's to fix.

### 4.2 The objective half, which the refiner cannot erase

`goal_adjusted_net` wraps the per-segment net in exactly the shape
`kairos.optimize.revenue_net.segment_net_revenue` has, so it threads into the
greedy step, the F1 refiner and the exact DP tier through the one `net_of`
parameter all three already share. All three then climb the same adjusted scalar,
so there is nothing to optimise back out.

```
adjusted(segment, k) = net(segment, k)
                     + shadow * pressure(day) * price(day) * points(segment, k)

points(segment, k) = k * baseline_tvr * (break_length_seconds / unit_seconds)
price(day)         = the day's own points-weighted mean CPP, from its segments
shadow             = 1.0, what a committed point is worth as a multiple of that
```

`price` is read from the segments in front of it, never from a rate nobody
supplied. It stands for what the channel would have to give away to make good a
point it committed to and missed. `pressure` is the same figure the ranking half
uses, so the term vanishes the moment the goals are met and the objective is the
untouched net again.

The term is proportional to POINTS while the net is proportional to SHEKELS. That
is the whole reason it moves anything.

---

## 5. The measurement

Operator channel, all 30 real broadcast days on disk, 2540 segments,
`objective_mode='revenue_net'`, `refine=True`, the shipped guardrails. Each day
carries one goal-based order sized as a share of that day's whole expected
rating. A is the run with no goal; B is the same run with the goal through the
seam.

| goal, as a share of the day's rating | segments moved | revenue | goal points |
|---|---|---|---|
| 0% | 0 of 2540 (0.00%) | +0.0000% | +0.0000% |
| 25% | 53 of 2540 (2.09%) | +0.1941% | **+0.3754%** |
| 60% | 116 of 2540 (4.57%) | +0.2625% | **+0.8320%** |

Read honestly:

* **The identity holds exactly.** A zero goal reproduces the plan to the shekel.
* **The goal moves the plan.** 53 and 116 segments carry a different break count
  because the engine can now see a commitment it could not see before.
* **Goal points rise, monotonically with the size of the goal.** That is the
  direction the seam is supposed to push, and it is the number the trader's
  commitment is denominated in.
* **Revenue also rises, which was not the intent.** The goal term lets the search
  reach counts the pure net search did not, and on this data those counts happen
  to be better on plain revenue too. It is a measured side effect on one month of
  one channel, not a claim that a goal is free. A larger goal on thinner
  inventory should be expected to cost revenue, and the product should say so
  when it does rather than be surprised.
* Every figure above is `refine=True` and `dp_refine` on, so it is the shipped
  search and not a weakened one.

---

## 6. The call sites the lead must land

Neither is in this piece's owned paths, and both are one line.

**A. The ranking half, into the single demand fold.** In
`kairos/service.py::_assemble_demand_weights`, replace the closing `return
build_demand_weights(...)` with the same call bound and folded:

```python
weights = build_demand_weights(
    segments, engine,
    inventory_weights=inventory_weights,
    pacing_weights=pacing_weights,
)
return goal_seam.fold_into_demand_weights(weights, segments, today)
```

plus `from kairos.optimize import goal_seam` at the top. This is the one fold
every optimize path reaches through `day_core._optimize_one_day`, so the live day
plan, the scenario slider and the weekly export all get it at once. It is proved
inert on today's data. Section 4.1 is the honest caveat: land it for correctness
and composition, not expecting it to move anything until the refiner reads it.

**B. The objective half, into the net-mode primitive.** In
`kairos/optimize/optimizer.py`, inside the existing `if net_mode:` block:

```python
_net_of = goal_seam.goal_adjusted_net(
    segment_net_revenue, originals, goal_seam.load_goal_orders(), pacing_today,
)
```

This is the half that measurably works. It needs a reference date reaching
`optimize_breaks`, which today stops at `day_core`. Two honest options for the
lead: thread the existing `pacing_today` one level further, or have `day_core`
build the wrapper and pass it in. The seam is indifferent to which.

Both call sites are gated by the same arithmetic identity: with no real
goal-based order on disk, `load_goal_orders()` is empty, `goal_adjusted_net`
returns the original function object, and `fold_into_demand_weights` returns its
input map. The golden cannot move.

---

## 7. What this piece did not reach

* **No dashboard surface.** The order kind and the feasibility read are on the
  API record and in both languages, and nothing renders them yet. The hook is
  `record["order"]` on every campaign from `campaigns_with_flights`, and
  `campaigns_goal_order.goal_orders_read()` for the pre-flight answer.
* **No route.** `kairos_api/campaigns_api.py` is not this piece's to edit. The
  read is a function, not an endpoint. One `@router.get("/campaigns/goal-orders")`
  returning `goal_orders_read()` lands it.
* **G1-c cannot be measured on real data.** The plan on disk covers 2024-11 on
  the operator channel; every stored goal's window is 2025-04 to 2025-05. So
  `expected_supply_per_day` honestly returns `None` for every stored order and
  every verdict is `unknown`. The read is exercised against supply in tests, and
  the unknown path is exercised against the real store.
* **G1-d is already true and was not rebuilt.** The pacing board already reads
  the goal as its denominator and the make-good ledger already settles a
  shortfall. Nothing here changed either, deliberately.
* **The refiner erasure is unfixed.** Section 4.1 is a finding about the whole
  demand-weight class. Fixing it means the group objective reading the weights,
  which is an engine internal this piece does not own.
