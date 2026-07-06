# Assistant capabilities: the right architecture, and the build

The assistant is the one surface where the product can use a model to reason
over its own real machinery. This document sets the architecture, decides the
open questions (including whether to build an MCP server), and specifies the
capabilities to build, grounded in the current Anthropic API surface.

## The architecture decision: in-process tool use, not MCP

The assistant is an in-app feature of one FastAPI backend. Its tools are the
backend's own Python functions: they read the same in-process caches, the same
optimizer, the same settings. They already run inside the request. The
question the owner raised is whether to expose them through an MCP server built
with FastMCP.

The answer is no, and the reason is specific. MCP is a protocol for crossing a
boundary: exposing tools to a separate client (Claude Desktop, another agent, a
different application) over a transport. Every one of those words describes a
cost the assistant does not need to pay. The tools live in the same process as
the engine; wrapping them in an MCP server would add a second process, a
serialization layer, a transport, and a second auth crossing, to reach
functions that are one Python call away. It would make the assistant slower and
more fragile for zero capability gain.

MCP is the right tool for a different goal: letting an EXTERNAL agent drive
Kairos (a Claude Desktop operator asking Kairos to optimize a day, a scheduled
cron agent running the weekly recompute, a partner integration). If that
becomes a goal, a FastMCP server that re-exposes the same tool registry is the
clean way to do it, and the registry below is deliberately shaped so it could
be lifted into one without change. It is a future surface, not a dependency of
making the in-app assistant more capable.

So the assistant stays on the Claude API with a manual agentic loop. The manual
loop (not the SDK tool runner) is correct because the assistant needs
fine-grained control the runner does not give: proposals are captured and never
executed, every step is audited, and applies are gated behind human approval
and a restore point. This is exactly the case the API guidance names for the
manual loop.

Model: claude-sonnet-4-6 (the newest Sonnet; there is no Sonnet 5). It supports
adaptive thinking and the effort control, which the goal-seeker below uses.

## Current state

kairos_api/assistant.py runs the manual loop; kairos_api/assistant_tools.py is
the tool registry (READ tools execute in-loop, PROPOSE tools are captured as
pending proposals); kairos_api/assistant_actions.py is the proposal store,
apply engine, restore points and audit log. Read tools today: get_settings,
get_day_detail, list_constraints, list_overrides, get_pricing,
get_net_comparison, get_compliance. Propose tools: settings change, constraint,
override, pricing change, recompute. The answer is grounded-only.

## The capabilities to build

Two land in this pass; the rest are specified so they compose onto the same
loop later.

### 1. In-chat simulation (the enabling primitive)

A READ tool, simulate_settings_change, that runs the owned-channel scenario
under a proposed set of settings WITHOUT applying anything, and returns the
before and after: gross, retention cost (with the calibrated band), net, and
breaks, plus the deltas. It reuses the exact scenario runner and the shared
owned-channel scope selector, so a simulated number is the same number the
plan would show. It writes nothing.

This turns every what-if into a free experiment: "what happens if I raise the
revenue weight to 70" gets a real before and after in the chat, with no risk
and no save. It is also the primitive the goal-seeker climbs.

### 2. Goal-seeker (the agentic payoff)

When the operator states a goal ("get me to a higher net without dropping
retention below 0.75"), the loop lets the model call simulate_settings_change
repeatedly, comparing outcomes, until it converges on a settings set that meets
the goal, then emits a normal proposal for it. Nothing is applied along the
way; the operator still approves the final proposal, and the restore point and
audit still apply. This is what makes the assistant an analyst rather than a
lookup: it uses the optimizer's own tools as an agent loop. The loop runs with
adaptive thinking and a raised iteration budget so it has room to search.

### 3. Provenance on every number (the trust layer)

Each read tool result is tagged with its source (which endpoint or dataset the
number came from), and the system prompt requires the answer to name that
source for each figure. The response surfaces the source trail so the operator
can see, for any number, where it came from. This is the discipline the whole
product sells: an answer that cites its own machinery, and never a figure
without a traceable origin.

### Specified for later (same loop, no new infrastructure)

- Streaming: stream the answer over Server-Sent Events so the goal-seeker's
  progress and the final text appear as they are produced rather than after a
  pause. FastAPI StreamingResponse plus the SDK stream helper; the loop shape
  is unchanged.
- Persistent per-operator memory: the conversation and the operator's stated
  preferences survive a restart (today the thread dies with the process). A
  small per-operator store, the same shape as the audit log.
- Proactive honest digest: a scheduled read-only pass that surfaces real
  movements ("net moved because X"), never a fabricated insight.

## The moments that carry the experience

Three moments are where this feels like more than a chatbot, and each is a
design commitment, not a feature:

- The what-if that costs nothing: a number changes in the chat and the operator
  did not touch the live plan. Simulation is a first-class, side-effect-free
  answer.
- The goal met by search: the operator states an outcome and watches the model
  try settings against the real optimizer and come back with a plan that hits
  it. The optimizer's own tools, driven as an agent loop.
- Every figure traceable to its origin: no number without a source, so the
  operator can trust the answer the way they trust the dashboard.

## What stays true

Grounded-only: the model reasons over real tool results, never invents a
number. Propose-and-approve: the model proposes, the operator approves, a
restore point is written first, everything is audited. The competitor boundary:
the assistant only ever reasons about and proposes actions on the owned
channel. The model is claude-sonnet-4-6.
