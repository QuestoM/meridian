# Kai: coverage gap, mention prior art, and a design

Researched 2026-08-07 against the installed binaries and recorded sessions, not documentation.

Binaries resolved, transcripts read. The evidence upgraded three claims and overturned my framing of one. Here is the revised report.

---

# Kai: coverage audit, mention prior art (binary- and transcript-verified), and a design

**What changed from my first pass:** the architecture question is now settled from local artifacts, not docs. Two findings are new and one of them changes the design's status from *invention* to *precedent*. The coverage half is unchanged — it was already measured from this repo's source.

---

## Part one — coverage gap (unchanged; measured from source)

Measured by importing `kairos_api.assistant_tools` in the repo venv: **31 read tools, 8 propose tools**. Context = 6 fingerprint-cached base sections + `per_day_plan` (always) + `day_detail <date>` (on a named plan date) + 12 keyword sections + `current_location`, under a 60,000-char budget.

"Vocabulary" = the prompt or handbook gives Kai the word without the data. That is the worst cell, because prompt rule 2 pushes the model to answer and rule 3 is the only brake.

| Surface / store | Read | Propose | Vocabulary | Evidence |
|---|---|---|---|---|
| **Break pod + per-spot arithmetic** (`break_api_pod*.py`) | **none** | **none** | **yes, and instructed** | Zero hits for `pod` in `assistant*.py`. Rule 24 names "the break, the spot"; the `traffic_operator` job voice says *"Answer in terms of breaks, spots and durations, and be exact about seconds."* Kai is told to talk in seconds about an object it cannot read. |
| **Pacing board** (`/api/pacing`, verdicts, trigger, remedy, `/accept`) | **none** | **none** | partial | Only `pacing_status`, which carries `flights_count` and nothing from the board. |
| **Make-good ledger** (`makegood_store.py`, `data/make_goods.csv`, `/api/make-goods`) | **none** | **none** | yes (rule 24) | `get_make_good_alerts` reads the *old* projection over `data/campaign_flights.csv`, which is **1 line — header only** (verified). It will answer `data_available:false` forever, while `data/make_goods.csv` (31 columns, written today) is unreachable. |
| **Campaign delivery, tri-state basis** (`campaigns_delivery.py`) | **none** | **none** | none | The module whose whole point is honest basis is invisible to the assistant whose whole point is honest basis. |
| **Campaigns / flights / commitments / assets** | none (a count) | none | partial | |
| **Model console: candidates + adoption** (`/candidates/*`, `/decisions`, `/gates`, `/training`, `/drift`) | **none** | **none** | none | Only `get_audience_model`, `get_audience_stability`, `model_state`. Adoption entirely absent. |
| **Licence limits & guardrails** (`guardrail_store.py`, attestation, effective date, change log) | **none** | **asymmetric** | yes | `assistant_permissions.guardrail_permission()` knows the four `GUARDRAIL_KEYS` and **refuses** proposals touching them — but no tool lists them. Kai can decline to change a number it cannot state. |
| **Plan versions** (`plan_version_store.py`) | none | none | yes, **and ambiguous** | Rule 24 says "the plan version"; the handbook's §Versions describes the *operation-state* store. One word, two objects, data for neither. |
| Operation-state versions / Target / Inventory | none | none | yes / yes / none | |
| **Product uploads & refusals** | partial | none | partial | `get_upload_status` gives validity, `in_use`, `engine_reads`, warnings. `uploads_remedy.py` / `uploads_messages.py` unreachable — Kai can say a file is invalid, not what the product told the operator to do. |
| Break board / states / pins | partial | `propose_override` | partial | |
| Scenario / week compare | partial | none | partial | |
| Accounts / roles / affiliation | none | n/a | yes | |
| **Plan, gold, yield, frontier, freshness, run log** | **full** | `propose_recompute` | full | |
| **Advertisers, agencies, events, pricing, constraints, overrides, settings, compliance, reports, activity, agreement uploads** | **full** | 7 tools | full | |

**Shape of the gap:** everything built recently is invisible, and four of the invisible things (pod, spot, make-good, plan version) are named in the prompt's own object vocabulary. *Words without data* is more dangerous than plain absence.

**Bug found in passing:** `assistant_page_context.ENTITY_TYPES` supports `program` and `_entity_program` is a complete builder, but **no surface calls `useAssistantEntity('program', …)`** — only `advertiser`, `agency`, `event`. The programme path is dead code.

---

## Part two — the current mention mechanism, measured

**There is none.** `AssistantComposer.jsx` is a plain `<textarea>`, `maxLength={2000}`; the only key handling is `AssistantPanel.jsx:246` (Enter sends, Shift+Enter newlines). Zero trigger characters, zero autocomplete.

What exists is **ambient, not addressed**: `page_context`. The shell publishes `{view: activeView, label}` at **rail-label granularity** (`TVBreakDashboard.jsx:305`) — not a tab, not a selected day, not a row. A page may register **one** focused entity of **four** declared types, of which **three** are ever emitted. Prompt rule 22 tells Kai to resolve `שלו` against it.

It is well built and the wrong shape: it carries *where you are*, never *what you mean*. You cannot refer to a second thing, a thing on another page, or anything below entity granularity. And the binding is invisible — nothing tells the operator that "it" resolved, or to what.

---

## Part three — prior art, from the binaries and the recorded sessions

### Provenance, stated plainly

| | Artifact | What it can and cannot prove |
|---|---|---|
| **Claude Code 2.1.212** | `/opt/homebrew/Caskroom/claude-code/2.1.212/claude` — Bun-compiled Mach-O with the **JS bundle readable as strings**. | Logic *and* strings are readable. Every quote below is verbatim from it. |
| **Codex 0.144.1** | `/Users/home/.codex/packages/standalone/releases/0.144.1-aarch64-apple-darwin/bin/codex` — **native Rust Mach-O, 260 MB**. | **Not a JS bundle. Compiled machine code — the logic is NOT greppable.** Only string literals are. So Codex's *logic* below is read from the open-source repo at `main`, and the installed binary is used to **verify version alignment** by string presence. I flag this rather than pretending I read the shipped logic. |
| **Codex sessions** | `~/.codex/sessions/` — 468 JSONL rollouts, **10,952 user turns**. | Shape only; no owner content quoted. |
| **Claude Code transcripts** | `~/.claude/projects/` — 31,165 JSONL. | Shape only. |

### Finding 1 (new, decisive) — Claude Code ships **two** file-reference shapes, and both are live

`attachment` is a persisted top-level record type in the transcript. Across 1,463 transcripts:

| `attachment.type` | count | key-set | content? |
|---|---|---|---|
| `file` | **1,732** | `content, displayPath, filename, type` | **yes — inlined** |
| `compact_file_reference` | **805** | `displayPath, filename, type` | **no content field at all** |
| `nested_memory` | 96 | `content, displayPath, path, type` | yes, + `contentDiffersFromDisk` |

The `file` attachment's inner payload, 1,732/1,732 with an identical key-set:

```
content.file = { filePath, content, numLines, startLine, totalLines }
```

`numLines` vs `totalLines` is the truncation accounting; `startLine` is where `@f.ts#L100-200` lands.

**This is the finding that matters most.** Claude Code does not have one mention architecture — it has an **eager** one that inlines content and a **lazy** one that carries only a path, and real usage is 1,732 to 805. The hybrid I proposed in my first pass is therefore **precedent, not invention.**

`nested_memory` carries **`contentDiffersFromDisk`** — a shipped boolean saying the in-context copy no longer matches the file. That is precedent for the `changed` state I proposed.

**What the transcript cannot show:** `isMeta` records are **not persisted** (0 across all 31,165 files). So the synthesized `Called the Read tool with the following input: …` / `Result of calling the Read tool: …` framing, wrapped in `<system-reminder>`, is a request-build artifact invisible in the log. **The transcript proves the attachment layer; only the bundle proves the message layer.** The bundle is unambiguous:

```js
case"text": return om([xen(nv.name,{file_path:e.filename}), Ren(nv,r), …])
function xen(e,t){return $r({content:`Called the ${e} tool with the following input: ${He(t)}`,isMeta:!0})}
```
`nv` = the Read tool; `om()`/`Zv()` wrap in `<system-reminder>`.

Per-kind divergence (bundle, VERIFIED): directories → flat **one-level, names-only** listing capped at 1000 with `… and N more entries` (the docs' "with file information" overstates it). MCP resources → `<mcp-resource server= uri=>`. **Agent mentions inject nothing** — docs confirm: *"The @-mention controls which subagent Claude invokes, not what prompt it receives."*

### Finding 2 (new) — Codex: the structured mention path is exercised **zero times in 10,952 turns**

The recorded `user_message` event carries exactly:

```
['type','client_id','message','images','local_images','audio','local_audio','text_elements']
```

Across all 468 sessions:

```
total user_message events : 10952
  text_elements null      : 0
  text_elements []        : 10952
  text_elements non-empty : 0
  with local_images       : 149
```

**Every single turn has an empty `text_elements`.** A heavy user, ~11k turns, and the structured-reference channel never fired once — while the image path fired 149 times. Two readings, both true: `text_elements` only ever carries plugin/skill bindings (files produce nothing structured, per the repo), *and* nobody reaches for it.

That is a design warning aimed at me, not at Codex, and I act on it below.

### Finding 3 (new) — the Codex binary confirms files are never read, by the absence of the machinery

If Codex inlined mentioned file content there would have to be a size or truncation notice for it. Strings present/absent in the shipped 0.144.1 binary:

| String | Count | Meaning |
|---|---|---|
| `exceeded the main prompt context limit and was truncated` | **1** | skills are inlined and capped (8,000 bytes) |
| `could not read the local` | **1** | local media read failure |
| any file-mention size/truncation notice | **0** | **files are never read, so nothing can be truncated** |
| `All Results` / `Filesystem Only` | 1 / 1 | the three search modes ship |
| `no matches` | 2 | picker empty state |
| `mentions_v2/render.rs` | present | the mention module ships; `mentions_v2` is default, flag is rollback only |
| `plugin://` / `skill://` / `app://` | 4 / 10 / 5 | the structured binding schemes ship |

Skills have a cap notice. Images have an error notice. Files have neither. Corroborated by the repo: `protocol/src/user_input.rs` has **no `File` variant** (`Text`, `Image`, `LocalImage`, `Audio`, `LocalAudio`, `Skill`, `Mention`), and `input_submission.rs` has **no branch that reads a mentioned file**.

### The two architectures, side by side

| | Claude Code 2.1.212 | Codex 0.144.1 |
|---|---|---|
| Buffer holds | plain string | plain string (files, **sigil discarded**) / atomic element + `{sigil, mention, path}` binding (plugins, skills) |
| Reaches the model as | **`file`: content inlined** at send. **`compact_file_reference`: path only.** Both ship. | **a bare path inside `ContentItem::InputText`** |
| Timing | send time | never resolved |
| Cost gate | 256 KiB / 25k tokens / 2000 lines; oversized **silently dropped** | none needed |
| Staleness signal | `contentDiffersFromDisk` (nested_memory) | none for text |
| Missing target | **silently dropped**, no marker to model or user | nothing (media only: `Codex could not read the local …`) |
| Drill into a container | **no** — accepting a directory appends a trailing space and dismisses; the space kills the token so the popup cannot reopen | **no** — `File` and `Directory` both map to `Selection::File(path)`, popup closes |

**Triggers.** Claude Code: `@` (files, dirs, `@server:uri`, `@agent-name`), `/`, `!` (shell), `:` (emoji, 2.1.217+). `#` is **not** a trigger — `function QO(e){if(e.startsWith("!"))return"bash";return"prompt"}` is binary. Codex: `@` (unified: files, dirs, plugins, skills), `/`, `$` (`TOOL_MENTION_SIGIL`, with real `$HOME`/`$1` shell arbitration), `!` (local shell, **never sent**). `#` is not a trigger in either.

**Search.** Claude Code: `git ls-files` index, cwd only, directories *synthesized* from ancestors; hand-written subsequence scorer, smart-case, +8 at `/ \ - _ . space`, +6 camelCase, shorter-path bonus, test-file demotion, 15 results, **no recency**. Codex: `ignore`-crate walk (hidden **included**), **nucleo** for paths + a hand-rolled matcher for tools; kind rank dominates (Plugin→Skill→File/Dir); cancellable (`AtomicBool` every 1024 entries, 10 ms tick, stale drop by session token *and* query equality); 20 fetched, **8 shown**.

**Rendering.** **Neither product uses icons.** Claude Code carries kind by text shape alone. Codex uses a right-aligned text tag column padded to `"Plugin"` width — `Plugin` (magenta), `Skill `, `File ` (cyan), `Dir ` — and resolves same-name ambiguity with a **two-column row: basename primary, dim parent path secondary**, match indices bolded in both.

### Screenshots — not taken, and why

`/usr/sbin/screencapture` exists. The only running `claude` processes are this session's own headless agents; no interactive TUI is displaying a picker. Getting a mention-picker screenshot requires opening an interactive session of either product, which I was told not to do. Capturing the screen now would capture the owner's desktop and zero evidence. **So every rendering claim above is sourced to code, and the visual gaps are listed at the end rather than filled by guessing.**

---

## Part four — the design for Meridian

### 4.1 Trigger: `@` only

Codex splits `@`/`$` because files and tools are disjoint namespaces. Meridian has **one** namespace of business objects. A second sigil costs a thing to teach traffic operators and buys nothing. `/` and `#` reserved, neither shipped. Dates get no sigil — `_question_dates` already parses ISO, `dd/mm`, `dd.mm` and Hebrew/English weekdays conservatively out of prose, and `@` addresses a broadcast day as well, so both routes reach the same object.

`@` is Shift+2 on the Hebrew layout, same key as English — no layout switch.

**Design consequence of Finding 2 (new).** 10,952 turns with zero structured mentions says a mention system that is the *only* way to refer to something will simply not be used. **So the free-text path must keep working exactly as it does today** — `_question_dates`, `find_advertiser`, the keyword sections. The mention is an accelerator and a disambiguator, never a gate. Nothing in this design removes a way of asking.

### 4.2 Taxonomy and icons

One glyph per **kind**, never per instance, from lucide (already the product's set). **Rule: where a kind has a rail destination, the chip's glyph is that rail item's glyph** (`shell/nav.js`) — the icon is navigational identity, not decoration. Both reference products ship *no* icons; this product should, because its kinds are heterogeneous business objects, not one kind (files) with a path that already discloses type.

| Family | Kinds | Icons |
|---|---|---|
| **Plan spine** | broadcast day · programme · break · pod · spot | `CalendarDays` · `MonitorPlay` · `ClipboardCheck` · `Layers` · `Film` |
| **Commercial** | campaign · flight · advertiser · agency · make-good | `FileBarChart` · `FileBarChart` dim · `Users` · `Building2` · `Gauge` |
| **Rules & money** | constraint · override · rate-card layer · licence limit · calendar event | `ListChecks` · `SlidersHorizontal` · `Layers` · `ShieldCheck` · `CalendarDays` outlined |
| **Provenance** | upload · plan version · model candidate · report | `Database` · `History` · `Bot` · `ListChecks` |
| **Scope** | channel (exactly one, always the operator's) | `Radio` |

Hebrew labels are **read from the existing words modules** (`pacing_alerts_api_words.py`, `campaigns_api_words.py`, `makegood_store_words.py`, `vocabulary.js`, prompt rule 24), never invented. A kind without an approved word does not ship until it has one.

### 4.3 Search, and the competitor boundary

**The candidate index is server-side.** The saved plan holds every channel because the retention model is measured against the competitive lineup (`assistant_context._owned_frame`). A client-side index puts rival rows in the browser. So: `GET /api/assistant/mentions?q=&types=&limit=` — one route, one enforcement point, reusing `channel_scope.scope_frame` and `_owned_frame()`.

Three boundary rules:

1. **The cap is applied after scoping.** An `omitted` count computed before scoping *is* a rival count — the decision `_section_counts` already documents ("How many rival rows were dropped is a fact about rivals").
2. **No-match and not-ours are indistinguishable.** Typing a rival channel's name returns the same "no matches" as a typo. Not "none on your channel" — that confirms the name exists.
3. **No kind is searchable whose store is not channel-scopable.** The pod is safe by construction: `break_api_pod`'s docstring records the traffic file has **no channel column at all**, and the channel shown beside a pod is the operator's own from settings.

**Matching** — take Codex's shape:
- Kind rank first, ordered for this product: exact id → day → programme → break/pod → commercial → provenance. A bare number usually means a time or a date.
- Within a kind: case-insensitive subsequence over label *and* id with Codex's prefix bonus (`if first_pos == 0 { score -= 100 }`) — ~20 lines.
- **Hebrew requirement neither product has:** strip the one-letter prefixes `ובלמהשכ` on both sides, or `בחדשות` never matches `חדשות`. **Reuse `assistant_context._strip_hebrew_prefixes`; do not write a second one.**
- Small corpus (~7 days, ~2,400 owned segments, tens of campaigns) → hold it in the existing fingerprinted `read_cache`, so a run invalidates it rather than leaving it stale.
- 20 fetched, **8 shown** (Codex's `MAX_POPUP_ROWS`), incremental with query-equality staleness drop.

**Row layout: Codex's two-column answer, unchanged.** Primary = the object's name; dim secondary = its **parent path in the hierarchy**. A break row reads `20:40` over `2024-11-01 · חדשות הערב`. That single decision solves the dominant ambiguity here — programmes that recur every weekday — which has no analogue in a code editor.

### 4.4 Drill-down — the part both products verifiably declined to build

Claude Code kills it with a trailing space; Codex collapses `Directory` into `File`. Both can afford to, because a flat fuzzy search substitutes for navigation *when every leaf has a unique typeable path*. **A spot has no name.** So this product must build what both declined.

One popup, two modes:
- **SEARCH** (default): flat ranked list across kinds — the 90% case, and both products are right that it suffices there.
- **DRILL**: the **leading-edge arrow** on a container row descends into its children; trailing-edge ascends; the header becomes a breadcrumb; Enter accepts at any depth. This is exactly the chaining Claude Code gives bash-path completion (`tuf`) and withholds from `@`.

**Arrow direction resolves from `documentDirection(locale)`, never hardcoded** — in RTL, leading is ArrowLeft.

The ladder is a **graph of typed edges, not a tree**:
```
channel → day → programme → break → pod → spot
advertiser → campaign → flight → spot
agency → advertiser
```
A spot is reachable by two ladders and is the same object either way. That is what a file tree cannot express, and it is why this is `GET /api/assistant/mentions/children?type=&id=&edge=` rather than a path walk.

**Honest empties in the drill.** `break_api_pod` records that on `רשת 13 / 2024-11-01` the plan's 80 breaks and the ledger's 72 pods sit a median 156 s apart, with only 21 of 72 within 60 s. A break often genuinely has no pod on disk. The drill shows a stated row — *this break has no pod in the traffic file* + reason — never an empty list, which reads as "zero spots".

### 4.5 The chip, and RTL

**A chip, not text.** Both products insert plain text for files; Codex makes an atomic element in exactly one place — plugins and skills, *where the referent has an identity that is not a path* — with a side-table binding and a `[@Name](plugin://x)` wire form so bindings survive a resume. **Every object here is that case.** So: chip for every kind.

Composer state becomes `{ text, refs: [{start, len, type, id, label}] }` — offsets into plain text, the plain text carrying the human-readable **label**, so the box still reads as a Hebrew sentence and copy-paste yields readable prose. Rendering is a highlighted overlay run with a glyph — **not** a rich-text editor. A contenteditable in an RTL dock is a large bidi-fragile build that buys nothing the overlay does not.

**How the chip uses `bidi.jsx`** — this is the whole bug class that file exists for:

- Label → **`<Name>`** (first-strong isolate): `Coca-Cola` stays Latin inside a Hebrew line, `חדשות הערב` stays Hebrew inside an English one. One call, correct in both locales.
- Identifier when shown (segment id, upload id, version hash) → **`<Code>`**.
- Any figure the chip or preview carries (time, duration in seconds, money, date) → **`<Figure>`**.
- The picker's dim parent path mixes a date and a name → a `<Figure>` and a `<Name>` **side by side**, never one `isolate()` around the concatenation. `isolate()`'s own docstring warns against wrapping a phrase.
- **The chip is an inline `<span>`.** `bidi.jsx` states it: isolation on a block re-anchors alignment. A chip built as a `div`, or carrying `dir`, would left-align the composer line in the Hebrew dock. The glyph is a sibling span *inside* the `<Name>`.
- **The picker popup is portalled** (MUI Popper), lands outside the shell subtree, inherits nothing → it gets **`<DirectionRoot locale={locale}>`**. That is exactly the fourth root case `bidi.jsx` enumerates, and the only place this design sets `dir`. `verify-direction-rules.mjs` catches any other.

### 4.6 How a mention reaches the model

**Resolve at SEND time, expand to a bounded typed card, deliver as a new CONTEXT section `mentioned_objects` — not as a synthesized tool call.**

Four reasons:

1. **Honest math forces it.** Prompt rule 2 requires every figure to name its section or tool; rule 19 requires the scope. Codex's bare path carries no basis. A `mentioned_objects` card carries `basis` and `scope_channel` per figure, exactly as `get_top_advertisers` and `_section_pacing_status` already do. **The mention must carry its basis, so it cannot be a bare identifier.** This single constraint rules out the Codex architecture outright.
2. **Cost is not the problem it is for files.** A break card is tens of fields, not 25k tokens. It rides the existing 60k-char budget with its own sub-budget and the existing drop policy (lowest-revenue-first, flagged `mentions_truncated`) — prompt rule 7 already covers that disclosure.
3. **Freshness.** Send-time, never insertion-time, and for a stronger reason than either product has: a *run* rewrites the plan under the operator. An insertion-time snapshot would let Kai quote a pre-run figure.
4. **Never fake a tool call.** Claude Code injects `Called the Read tool …` because its model already knows Read. Kai's trace is an **operator-visible audit surface** (`AssistantRunTrace.jsx`), and `kai-claimed-action.js` treats `tool_trace` as *the authority* on what happened — its whole existence is a defence against the model claiming an act it did not perform. Injecting a synthetic call corrupts the one artifact that proves what Kai did. Mentions ride in CONTEXT with their own source stamp; the trace records real calls only.

**Container mentions: eager for the thing, lazy for its contents.** `@day` expands to the day's own card **plus a child summary** — counts and the top few by revenue with the true total beside the cap (the `_cap` idiom already in the codebase). Full contents come from a real read tool if the model wants them.

**This is now precedent, not invention.** Claude Code ships exactly this split — `file` (content, 1,732 uses) and `compact_file_reference` (path only, 805 uses) — and both are live in real transcripts on this machine. The design decision is which shape each *kind* gets, and the answer here is: the mentioned object eager, its children lazy.

### 4.7 Degradation — stale or gone

Both products are weakest here, verifiably. Claude Code **silently drops** and the `@path` survives as prose with no marker; three of the last month's changelog entries (2.1.216, 2.1.217, 2.1.221) are bugs of exactly that shape, because the failure is invisible by design. Codex re-validates nothing for text.

Silent drop is **forbidden here**: Kai would see a Hebrew label in the question with no data behind it, and rule 2 would push it to answer from the label. That is fabrication.

Four states, all four reaching the model:

| State | Card | Chip |
|---|---|---|
| `resolved` | figures + `basis` + `scope_channel` + `as_of` | normal |
| `changed` | the **current** figure plus a note it moved since insertion | subtle marker |
| `gone` | `{status:"gone", type, id, label, reason}`, no figures | struck through, warning |
| `unavailable` | store unreadable; **distinct from `gone`** | warning |

`changed` has shipping precedent: Claude Code's `nested_memory` attachment carries **`contentDiffersFromDisk`**, a boolean saying the in-context copy no longer matches the source. `gone` vs `unavailable` is this product's own tri-state doctrine (`campaigns_delivery`'s aired/scheduled/unknown; `assistant_model_disclosure`'s "real, unavailable, unknown, never a confident guess"). A blank or zero in either cell is the exact fabrication those modules exist to prevent.

Add **prompt rule 31** naming `mentioned_objects` and the four states, mirroring rules 3 and 7.

**Ambiguity after insertion: nothing to resolve.** The chip carries typed `{type, id}`, so two same-named things are two different chips. Both products get this free from paths; here it comes from the id, and the picker's dim parent path resolves it at choice time.

---

## Part five — build order

**R1 — `@` over the four kinds that already resolve.** Advertiser, agency, event, **programme**, reusing `assistant_page_context._ENTITY_BUILDERS` verbatim, widening the frozen `page_context` contract from one `entity` to a list. The resolution path, boundary and honesty markers exist and are tested; the work is the picker, the chip and the plural. Fixes the dead `_entity_program` path for free.
*Smallest shippable. First time an operator can point at two things at once.*

**R2 — the mention route:** typed search, Hebrew prefix-stripped matching, two-column parent-path rows. *First thing that feels like Codex.*

**R3 — the plan spine and DRILL mode:** day and break kinds; `day → programme → break`; container cards with capped child summaries; honest empties. *Largest single feel change, and the thing no code editor has.*

**R4 — pod and spot. Coverage first.** Ship `get_break` and `get_pod` read tools **before** the mention kinds. A mention resolving to a card Kai cannot follow up on is a dead end, and the pod is the widest gap in the table.

**R5 — the commercial ladder** (`advertiser → campaign → flight → spot`) plus make-good, gated on new read tools for the pacing board and `makegood_store`. Note: **keep and relabel** `get_make_good_alerts` rather than repointing it — it is the signal the optimizer's pacing weights read, and its `data_available:false` is honest, not broken.

**R6 — provenance kinds** (upload, plan version, model candidate, report). Requires first resolving the plan-version / operation-state-version vocabulary collision, which is a handbook fix, not code.

---

## What remains unverified, and what would settle it

1. **Codex's shipped logic was not read.** The installed binary is native Rust — machine code, not a greppable bundle. Logic comes from the open-source repo at `main` (`85e0661c`); the binary verifies version alignment via string presence only (`mentions_v2/render.rs`, `All Results`, `Filesystem Only`, `no matches`, `plugin://`, `skill://`, `app://` all present in 0.144.1). *Settled by:* disassembly, or capturing the `sendUserTurn` payload off `codex app-server`.
2. **The Codex VS Code extension is closed source.** Whether an IDE mention is a clickable chip, and whether the IDE inlines content, is unknown. Issue #27749 asks for exactly the "context chip" UI, which is *evidence it does not exist* — but that is inference. *Settled by:* inspecting the shipped VSIX or the app-server socket.
3. **Whether Codex cloud/web supports `@` at all** — zero documentation either way.
4. **No mention-rendering screenshots.** Getting them requires an interactive session of either product, which I was told not to open. Every rendering claim is code-sourced. Specifically unconfirmed visually: how Claude Code renders a picker row mid-type; whether Codex's tag column is right-aligned in a narrow terminal; what either shows for a referent that vanished (the code says *nothing at all*, and I could not watch that happen). *Settled by:* one interactive session each, deliberately mentioning a file and then deleting it before sending.
5. **No real @-mention exists in either local corpus.** Codex: 0 non-empty `text_elements` in 10,952 turns. Claude Code: no at-mention `file` attachment traceable to a user mention rather than a tool read. So the transcripts confirm the *attachment shapes* and the *usage frequency*, and cannot confirm the insertion-to-send path end to end. *Settled by:* one deliberate mention in each product, then re-reading the newest transcript.
6. **Symbol mentions in Codex** — concluded absent from absence across source, docs, changelog and binary strings, not from a positive statement.

**Files that matter most for implementation:** `/Users/home/Code/questo/meridian/kairos_api/assistant_page_context.py` (the resolution path to widen), `/Users/home/Code/questo/meridian/kairos_api/channel_scope.py` (the boundary seam), `/Users/home/Code/questo/meridian/kairos_api/assistant_context.py` (`_strip_hebrew_prefixes`, reuse for matching), `/Users/home/Code/questo/meridian/kairos_api/assistant_sections.py` (`compose_context`, where `mentioned_objects` attaches), `/Users/home/Code/questo/meridian/tv-break-dashboard/src/shell/bidi.jsx`, `/Users/home/Code/questo/meridian/tv-break-dashboard/src/kai/AssistantComposer.jsx`.

**Caveat on the coverage table:** `pacing_alerts_api*.py`, `makegood_store.py`, `break_api_pod*` and their frontends are modified in the working tree by a live workflow. Those rows reflect the tree as I read it and may move.