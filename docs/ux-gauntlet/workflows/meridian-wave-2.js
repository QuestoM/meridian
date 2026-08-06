export const meta = {
  name: 'meridian-wave-2',
  description: 'Wave two: the four dependent pieces, with P13 chained behind P10 because its files live inside P10 tree, then an integration critic that cannot start before every piece has returned',
  phases: [
    { title: 'Build', detail: 'P10, P11 and P12 concurrently; P13 starts the moment P10 returns' },
    { title: 'Critique', detail: 'a fresh blind critic per piece per round, measuring the real product' },
    { title: 'Verify', detail: 'one integration critic over the whole wave, after the barrier' },
  ],
}

const LAWS = `
HARD LAWS (a violation fails the round):
- ultrathink. Recon before editing: read every file you touch in full, plus its tests, before any edit.
- Repo root /Users/home/Code/questo/meridian. Python ~/.venvs/meridian/bin/python. Frontend build: npm run build in tv-break-dashboard.
- OWNERSHIP IS ABSOLUTE. Write only the paths your piece owns in the wave-two table of section 8.2 of docs/ux-gauntlet/spec.md, plus your reserved test prefix and your declared helper modules under the section 8.2 naming rule. Every other path is FROZEN: read it, never write it. If you cannot finish without a frozen path, stop and report it as a blocker.
- Wave zero and wave one published contracts under docs/ux-gauntlet/contracts/. Read the ones for the modules you consume BEFORE you code against them. A frozen module public surface is in its contract; if the accessor you need is missing, report it rather than reaching into the module.
- Do NOT run git add, commit, push, stash, checkout or restore. Reading history is allowed and encouraged.
- Do NOT restart or kill the servers on ports 8000, 3000 or 8010. Start your own on the port your brief gives you and kill only that one.
- HONEST MATH: no screen may show a number that was not computed from real data. Missing capability is an honest empty state naming what is missing and the path to supply it, never a placeholder figure. Tri-state everywhere: real, unavailable, unknown.
- THE COMPETITOR BOUNDARY: the operator owns exactly one channel, read from settings. No rival channel name or figure reaches an operator surface or the assistant context.
- No em-dashes, no emojis, no exclamation marks in code, comments or copy. Sentence case. Files under 450 lines. One display string per source line, never hard-wrapped.
- Hebrew vocabulary canonical: ברייק, ברייקים, נעיצה, ברייקי זהב, רצועת שידור, הכנסה צפויה, עלות שימור, מפעיל, never משתמש. Israeli week: Sunday to Saturday, weekend Friday and Saturday, data ISO-keyed, presentation Sunday-first.
- TRAINING VERSUS RUNS: an act is training if and only if its output lands under models/. No operator surface may offer one. This is the line wave two is most able to blur, because P12 is a training piece.
- Your final message is data for the lead: what you built, exact paths, the measured evidence that each bar is met, and anything you could not do and why.

THE ASSISTANT NOW RUNS ON claude-opus-5. kairos_api/assistant.py DEFAULT_MODEL is claude-opus-5, overridable by KAIROS_ASSISTANT_MODEL. It takes adaptive thinking and output_config effort, and it REJECTS both budget_tokens and temperature. If you touch an Anthropic request builder, do not reintroduce either.

TWO DATA BLOCKERS ARE REAL AND APPROVAL CANNOT LIFT THEM. Do NOT invent around them.
- There is no current broadcast week, no populated campaign flights and no delivery feed. This blocks the delivered half of P11. Pacing against a plan that has no delivery is an honest empty state naming the missing feed, never a guessed delivered figure.
- There is no media feed and no asset durations from the owner. This blocks the read-a-video half of P13. The arithmetic verification (declared seconds against the sum of the spots in the pod) is real today and is what P13 builds. Do not fabricate a codec, an aspect ratio, a loudness figure or a duration you did not read.
A fabricated target, a guessed delivery figure or an invented media property fails the round outright.
`

// A sweep finds several things and the loop used to keep one. Measured on the
// wave-one close: P8's critic reported one gap and left three more in a field
// it named "not the gap but worth the next round", which nothing ever read. The
// sweep that found them was already paid for; discarding them bought a second
// full sweep to find a different one. So the verdict carries the whole queue,
// and every item says whether closing it is mechanical or structural, because
// that is what decides who closes it.
const FINDING = {
  type: 'object',
  additionalProperties: false,
  required: ['what', 'where', 'kind', 'severity'],
  properties: {
    what: { type: 'string', description: 'the finding, concrete enough to close in one pass' },
    where: { type: 'string', description: 'file and line, or the screen and the element' },
    kind: { type: 'string', enum: ['mechanical', 'structural'], description: 'mechanical: closeable by editing files this piece already owns, with no new endpoint, no new store and no data-model change. structural: anything else.' },
    severity: { type: 'string', enum: ['blocking', 'serious', 'minor'], description: 'blocking means the piece cannot pass with this open' },
  },
}

const VERDICT = {
  type: 'object',
  additionalProperties: false,
  required: ['passed', 'largest_gap', 'queue', 'fixed_myself', 'evidence', 'regression_verdict', 'previous_gap_closed', 'stopwatch', 'blind_ab'],
  properties: {
    passed: { type: 'boolean', description: 'true only when every bar is met with evidence you gathered yourself AND no blocking or serious finding remains open' },
    previous_gap_closed: { type: ['boolean', 'null'], description: 'null on round 1; otherwise whether each finding the last round left open is genuinely closed' },
    largest_gap: { type: 'string', description: 'the single largest remaining gap, concrete enough to fix in one pass; empty when passed' },
    queue: { type: 'array', items: FINDING, description: 'EVERY finding this sweep produced, ranked, including the ones you fixed yourself and the small ones you would otherwise have discarded. Never truncate this to the top one.' },
    fixed_myself: { type: 'string', description: 'the mechanical findings you closed yourself this round and the exact paths you edited, or the word none. A fresh judge reads only this diff.' },
    stopwatch: { type: 'string', description: 'Bar 1: the job stories you ran, with seconds, clicks, screens and dead ends, against the recorded baseline' },
    blind_ab: { type: 'string', description: 'Bar 2: the reference you opened, what you compared, and which won on each dimension' },
    regression_verdict: { type: 'string', description: 'Bar 3: what you verified still works exactly as it did before' },
    evidence: { type: 'string', description: 'paths to screenshots, test output, measured numbers' },
  },
}

// The scoped judge. It reads one diff against one list of claims. It cannot
// pass the piece and it is not asked to: passing still takes a full blind sweep
// on the stronger model. That bound is exactly what makes a cheaper model
// defensible here, against the standing rule that verifiers run on opus. The
// hole it could leave, a false green on a finding marked closed, is closed by
// the next full sweep, which is told to re-check every one of them.
const SCOPED = {
  type: 'object',
  additionalProperties: false,
  required: ['closed', 'reopened', 'collateral', 'evidence'],
  properties: {
    closed: { type: 'string', description: 'the findings the diff genuinely closes, each with what you measured to confirm it' },
    reopened: { type: 'string', description: 'any finding claimed closed that is not, with the measurement that shows it, or the word none' },
    collateral: { type: 'string', description: 'anything the diff broke that was working, or the word none. Run the tests adjacent to the changed files.' },
    evidence: { type: 'string', description: 'commands run, screenshots, measured numbers' },
  },
}

const INTEGRATION = {
  type: 'object',
  additionalProperties: false,
  required: ['suite', 'ownership_violations', 'seams', 'fabrication', 'largest_gap', 'evidence'],
  properties: {
    suite: { type: 'string', description: 'the exact pass/fail counts you measured for tests/, and how they compare to the wave-one close' },
    ownership_violations: { type: 'string', description: 'any path written outside its piece row, named exactly, or a statement that there were none' },
    seams: { type: 'string', description: 'whether the four pieces actually compose: P10 into P3 break surface, P13 into P10 pod, P12 against P7 releases, P11 against P4 clients' },
    fabrication: { type: 'string', description: 'any number on any screen not computed from real data, especially around the two data blockers' },
    largest_gap: { type: 'string', description: 'the single largest remaining gap across the wave' },
    evidence: { type: 'string', description: 'commands run and their output' },
  },
}

const PIECES = {
  P10: { model: 'claude-sonnet-5', effort: 'high',
    id: 'P10', name: 'Break contents and tonight breaks', port: 8041,
    stories: 'JS-6 and JS-7, the traffic operator who builds the pod and verifies it spot by spot',
    reference: 'a professional broadcast traffic or playout log, and Figma for direct manipulation of an ordered list',
    brief: 'You take break_api.py and src/plan/break/** by HANDOVER from P3, which froze at the end of wave one. Read P3 contract under docs/ux-gauntlet/contracts/ before you code. The break is an entity now: the pod is the ordered list of spots inside it, each with an advertiser, a declared duration and a position. Build the pod so a traffic operator can see it, reorder it, and read the arithmetic: declared break length against the sum of its spots, with the gap or overflow stated in seconds.' },
  P11: { effort: 'high',
    id: 'P11', name: 'Pacing and make-good', port: 8042,
    stories: 'JS-8, the account manager who must know a campaign is behind before it is too late to fix',
    reference: 'Google Ads for pacing against a flight, Stripe for how a shortfall states its remedy',
    brief: 'You own pacing_alerts_api.py, kairos_api/makegood_store.py (new) and data/make_goods.csv (new). Decision 4 is a REAL data blocker: there is no current broadcast week, no populated flights and no delivery feed, so the delivered half of pacing has no input. Build the make-good ledger, which is real and needs no delivery feed, and ship the pacing view as an honest empty state that names exactly which feed is missing and the path to supply it. Do not guess a delivered figure.' },
  P12: { effort: 'high',
    id: 'P12', name: 'Model improvement', port: 8043,
    stories: 'JS-19, the model steward deciding whether a candidate is genuinely better than what is live',
    reference: 'the best experiment-tracking or model-registry dashboard you can reach',
    brief: 'You own scripts/adopt_candidate.py (new) and models/candidates/**, and take models/releases/** by HANDOVER from P7. Read P7 contract first. This is the ONE piece whose purpose is training, so the training-versus-runs line is yours to hold precisely: adoption writes under models/ and is a company-only act, never reachable from an operator surface. Re-measure each candidate held-out gates, compute what would move, and let a steward adopt or reject with the verdict recorded in the artifact metadata.' },
  P13: { model: 'claude-sonnet-5', effort: 'high',
    id: 'P13', name: 'Media verification', port: 8044,
    stories: 'JS-7, the traffic operator verifying every spot in the pod is actually airable',
    reference: 'Frame.io for per-asset status at a glance',
    brief: 'You own kairos_api/media_api.py (new), kairos_api/media_store.py (new), data/media_assets.csv (new) and src/plan/break/media/**. That frontend path sits INSIDE P10 tree, so P10 has already finished and frozen before you start: read what it built and extend it, never restructure it. The owner media feed does not exist, so format, aspect ratio, loudness and codec are UNKNOWN and must read as unknown. What is real today is the arithmetic: declared spot seconds against the pod total. Build that verification and state every unreadable property as an honest unknown with the path to supply it.' },
}

// The durable-state contract. Wave one lost work three times to a limit, and
// the loss was never the file edits, which survive on disk: it was the thread
// between agents. The gap a critic named was interpolated into the next
// builder's prompt, so it lived only in this process's memory, and a process
// that dies takes it with it. Worse, because the gap text was part of the
// prompt, it was part of the resume cache key, so one lost critic changed the
// hash of every agent downstream of it and forced them all to run again.
//
// So the thread moves to disk. A critic WRITES its verdict to the state file
// and also returns it; a builder is TOLD WHERE TO READ IT rather than handed
// its contents. The prompt then varies only by piece and round number, both of
// which are deterministic, so the cache key is stable and a killed run resumes
// exactly where it stopped. The filesystem is the only thing in this system
// that outlives the process, so it is where the state belongs.
const STATE = 'docs/ux-gauntlet/state'

const STATE_RULES = `
DURABLE STATE, read this before anything else:
- Your piece has a state file at ${STATE}/<PIECE>.json. It is the memory of this
  loop and it is the only memory that survives a killed run.
- BEFORE YOU BUILD: read it. It carries the round number, whether the piece has
  passed, the gap named last round, and what previous rounds already did. If the
  file says work was already done, GO LOOK AT THAT WORK ON DISK before writing
  anything. A run that died mid-round still left its edits behind, and building
  the same thing twice is how a resume turns into damage.
- AFTER YOU FINISH: write it, with mkdir -p ${STATE} first. Keys: piece, round,
  passed, gap, history (append one short line per round, never rewrite past
  lines), files_touched.
- Write it even if you are about to fail or run out of room. A half-finished
  round that recorded what it did is recoverable; a finished round that recorded
  nothing is not.`

// One piece: the builder-critic loop. Returns the last round record.
// The critic is only ever spawned against an artifact a builder actually
// produced: agent() returns null when a run is skipped or dies on a terminal
// error after retries, and a critic handed a null build measures nothing while
// costing a full round.
async function runPiece(piece, inheritedNote) {
  let unclosed = 0
  let last = null
  let lostVerdicts = 0
  for (let round = 1; round <= 6; round++) {
    // The gap itself is deliberately NOT interpolated here. Naming the file
    // keeps this string identical for a given piece and round, which is what
    // makes the resume cache hold across a limit.
    const carry = round > 1
      ? `\n\nROUND ${round}. A blind critic measured your previous round and it did not pass. Its full ranked queue is in ${STATE}/${piece.id}.json under "queue". Read that file first, in full.
CLOSE EVERY ITEM MARKED structural, not only the largest one. They were all found by one sweep that has already been paid for, they are all named with a file and a line, and closing them one per round buys a fresh sweep for each. Work down the queue by severity: blocking first, then serious, then minor.
DO NOT REDO the items marked mechanical. The critic closed those itself and a judge has already read that edit. Touching them again only risks undoing a fix that held.
If the file is missing or its round is behind this one, the previous round died before recording: inspect the working tree for what was already changed for this piece and continue from there rather than starting over. Do not restructure what already passed.`
      : `\n\nROUND 1. Nothing has been built for this piece yet.${inheritedNote || ''}`

    const built = await agent(`You are the builder for piece ${piece.id}, ${piece.name}, in wave two of the Meridian experience rebuild.
${LAWS}
${STATE_RULES}
YOUR STATE FILE: ${STATE}/${piece.id}.json
READ FIRST, IN FULL: docs/ux-gauntlet-prompt.md (the campaign brief), docs/ux-gauntlet/spec.md sections 3 through 8 (what this piece is, who it serves, the training-versus-runs rule, and your wave-two ownership row), docs/ux-gauntlet/job-stories.md for ${piece.stories}, docs/ux-gauntlet/discovery/06-baseline.md for today's measured numbers, and docs/ux-gauntlet/three-way.md for what wave one changed.
YOUR PIECE: ${piece.name}. It serves ${piece.stories}. Its reference bar is ${piece.reference}.
${piece.brief}
FOUR BARS, ALL OF THEM:
1. STOPWATCH. The job stories this piece serves must be completable within their targets, measured in a browser. Build so a stopwatch would pass, not so a screenshot would look good.
2. BLIND A/B. A critic will open ${piece.reference} and compare it to yours without knowing which is which. Win on the mechanics, not the decoration.
3. REGRESSION. Nothing that works today may get worse, including everything wave one just built. This is a rebuild of a living system.
4. THE LAWS. Honest math, the competitor boundary, the vocabulary, the Israeli week, RTL correctness, file size, and the training-versus-runs rule.
If you need a running app, start your own on port ${piece.port} from the repo root and kill only that one when done: KAIROS_AUTH_DISABLED=1 nohup ~/.venvs/meridian/bin/python -m uvicorn kairos_api.server:app --host 127.0.0.1 --port ${piece.port} --log-level warning > /tmp/w2-${piece.id}.log 2>&1 &
Run the tests adjacent to what you touch and leave them green. Do not run the whole suite; the lead runs it at wave close.${carry}`,
      { label: `build:${piece.id}#${round}`, phase: 'Build', model: piece.model, effort: piece.effort })

    if (!built) {
      log(`${piece.id} builder produced nothing on round ${round}; not spending a critic on it`)
      return { piece: piece.id, name: piece.name, round, stopped: 'builder returned nothing' }
    }

    const verdict = await agent(`You are a BLIND CRITIC for piece ${piece.id}, ${piece.name}, round ${round}, in wave two of the Meridian experience rebuild. You did not build it and you must not defend it. ultrathink.
${LAWS}
YOU MEASURE FIRST AND YOU CLOSE THE SMALL THINGS YOURSELF. Measure every bar before you edit anything at all: an artifact you have already touched is no longer the artifact the builder shipped, and your measurement of it is worthless. Once every number is recorded, you may edit product code, under two hard limits. First, ONLY paths in this piece's wave-two ownership row of section 8.2. Second, ONLY findings you classified mechanical: a label, a missing disclosure, a wrong format or unit, a contrast or focus token, a hard-wrapped string, a dead end that needs a link, a copy fix. The moment a fix needs a new endpoint, a new store or a data-model change it is structural, and structural findings are NOT yours to close. Leave them for the builder and say so.
Why you and not a builder: you already have the file, the line, the screenshot and the number. A fresh builder has none of that and must spend a whole context rediscovering what you just measured. On the wave-one close, three of seven builder rounds produced no code at all for exactly that reason.
REPORTING MORE IS ALWAYS BETTER THAN REPORTING LESS, and your own fix load can never grow because of it: the mechanical ones are quick and the structural ones are not yours. Do not shrink a finding to mechanical to look thorough, and do not inflate one to structural to avoid the work. Both corrupt the queue. A structural finding you cannot close is a success of this round, not a failure of it.
You may also start your own server on port ${piece.port + 20} and kill only that one, write scratch files under /tmp, drive a browser, and read anything.
BEFORE YOU RETURN, WRITE YOUR VERDICT TO ${STATE}/${piece.id}.json (mkdir -p ${STATE} first): piece, round ${round}, passed, queue (EVERY finding, ranked, each with what, where, kind and severity), fixed_myself (what you closed and the exact paths), gap (the largest still open, the same words you return), history (append one line for this round, never rewrite earlier lines), and the evidence paths. Write it BEFORE you compose your final answer, not after. A limit that kills you between the two loses your judgment for good, and the next builder then rebuilds blind against a gap nobody recorded. The file is the durable copy; your return value is the convenient one.
THE QUEUE IS THE POINT. Every small thing you notice goes in it, including the ones that are not the largest gap and the ones you already fixed. A sweep is the expensive part of this loop and it has already been paid for by the time you notice the fourth thing. Discarding it means buying a second sweep to find it again.
You are deliberately not given the builder's reasoning. Judge the artifact by measuring it.
THE PIECE: ${piece.name}, serving ${piece.stories}. Its reference bar is ${piece.reference}.
MEASURE ALL FOUR BARS AND GATHER EVERY NUMBER YOURSELF:
1. STOPWATCH. Open the running product in a browser and actually perform each job story as that person would. Record seconds, clicks, keystrokes, screens traversed, dead ends, and every moment you had to guess. Compare against the target in job-stories.md AND against the baseline in 06-baseline.md. A piece that is prettier but slower fails.
2. BLIND A/B. Open ${piece.reference}. Capture both, compare on the mechanics that matter for this job, and say which wins on each dimension and why, concretely. If a reference is unreachable, say so plainly and lean on the stopwatch; never invent a comparison you did not run.
3. REGRESSION, the floor. Verify wave one capabilities on the surfaces this piece touches still work. Name each one and how you checked it.
4. THE LAWS. Any fabricated number, any rival channel name on an operator surface, any training action reachable by an operator, em-dashes, emojis, exclamation marks, files over 450 lines, hard-wrapped display strings, the word משתמש, Monday-first weekday order, a dead end where a name or a number should open something.
THE TWO DATA BLOCKERS ARE YOUR SHARPEST TEST. There is no delivery feed and no media feed. Any delivered figure, any pacing percentage, any codec, aspect ratio or loudness reading that is not sourced from a real file is a fabrication and an automatic fail. An honest unknown that names the missing input is a pass.
Also check ownership: run git status and git diff --stat and verify every path this piece changed is in its wave-two row of section 8.2. A write outside the row is an automatic fail.
Then return the full ranked queue, name the SINGLE largest gap still open after your own fixes, and set passed only when all four bars are genuinely met on your own evidence and no blocking or serious finding remains. Be willing to pass a piece that is done; a critic that never passes is as useless as one that always does.${round > 1 ? `\nPRIOR ROUNDS ARE IN ${STATE}/${piece.id}.json. Read the whole file. Two things there need you specifically. First, the queue the last sweep left open: state item by item whether each is now closed. Second, any finding a scoped judge marked closed: a scoped judge sees one diff and cannot see what that diff cost elsewhere, so re-check each of those yourself and reopen it if it did not hold. If the file is missing or its round is behind ${round}, say so plainly and judge the artifact on its own terms rather than inventing what the last gap must have been.` : ''}`,
      { label: `critic:${piece.id}#${round}`, phase: 'Critique', schema: VERDICT })

    last = { piece: piece.id, name: piece.name, round, built: String(built).slice(0, 2500), verdict }
    if (!verdict) {
      // A lost verdict used to end the piece here, which threw away a whole
      // round of real building because the judgment of it went missing. The
      // critic writes its verdict to the state file before returning, so the
      // next builder can still read what this one found. Carry on and let it.
      // Two losses in a row means the limit is not letting critics finish at
      // all, and continuing would just burn builders blind.
      lostVerdicts++
      log(`${piece.id} round ${round}: verdict lost in transit; the next round reads the state file`)
      if (lostVerdicts >= 2) {
        log(`${piece.id} stopped: two verdicts in a row never came back`)
        return { ...last, stopped: 'two verdicts lost in a row' }
      }
      continue
    }
    lostVerdicts = 0

    // The critic closed the mechanical findings itself. It must never be the
    // agent that clears its own edit, so a fresh judge reads that diff and only
    // that diff. Cheap by construction: one diff against one list of claims,
    // with no sweep, no browser tour of the whole piece and no reference
    // comparison. That narrowness is also its weakness, which is why it cannot
    // pass the piece and why the next full sweep re-checks whatever it cleared.
    const fixedAny = verdict.fixed_myself && String(verdict.fixed_myself).trim().toLowerCase() !== 'none'
    let scoped = null
    if (fixedAny) {
      scoped = await agent(`You are a SCOPED JUDGE for piece ${piece.id}, ${piece.name}, round ${round}, in wave two of the Meridian experience rebuild.
${LAWS}
A blind critic measured this piece, then closed the small mechanical findings itself rather than spending a whole builder round on a label. You judge THAT EDIT AND NOTHING ELSE. You did not make it and you must not defend it.
DO NOT re-review the piece. Do not run the job stories, do not open the reference product, do not sweep for new findings. Another critic does all of that next round and duplicating it here is the waste this judge exists to remove.
WHAT TO READ: ${STATE}/${piece.id}.json, specifically "fixed_myself" (what the critic claims it closed and where) and "queue" (the findings those edits map to). Then the edit itself: git diff and git status for exactly the paths named there.
WHAT TO DECIDE, on your own measurement and not on the critic's word:
1. Does each edit genuinely close the finding it claims, or does it only make the symptom less visible. A label that now reads correctly on one screen and not on the other is not closed.
2. Did the edit break anything adjacent. Run the tests next to the changed files. If a screen changed, look at it.
3. Is every changed path inside this piece's wave-two ownership row of section 8.2, and does every law still hold on the changed lines: honest math, the competitor boundary, no rival channel name, the vocabulary, RTL, no em-dash, no emoji, no exclamation mark, no hard-wrapped display string, files under 450 lines.
You are READ-ONLY. You may not edit any file. Report what genuinely closed, what did not, and any collateral. Reopening something is a useful answer, not a hostile one.`,
        { label: `judge:${piece.id}#${round}`, phase: 'Critique', model: 'claude-sonnet-5', effort: 'medium', schema: SCOPED })

      const notClean = (v) => v && String(v).trim().toLowerCase() !== 'none'
      if (!scoped) {
        log(`${piece.id} round ${round}: scoped judge lost; next sweep re-checks the critic's own edits`)
      } else if (notClean(scoped.reopened) || notClean(scoped.collateral)) {
        // The critic's own fix did not hold. Passing now would ship an edit
        // nobody cleared, so the piece goes round again whatever the critic
        // concluded about the rest of it.
        log(`${piece.id} round ${round}: judge reopened the critic's own fix: ${String(scoped.reopened || scoped.collateral).slice(0, 110)}`)
        last = { ...last, scoped }
        continue
      }
      last = { ...last, scoped }
    }

    if (verdict.passed) {
      log(`${piece.id} ${piece.name} passed on round ${round}`)
      return last
    }
    if (round > 1 && verdict.previous_gap_closed === false) {
      unclosed++
      if (unclosed >= 3) {
        log(`${piece.id} stopped: three rounds failed to close the named gap`)
        return { ...last, stopped: 'three rounds failed to close the named gap' }
      }
    } else {
      unclosed = 0
    }
    // No gap variable is carried between rounds any more. The critic has
    // already written it to the state file, which is what the next builder
    // reads, so holding a second copy in memory would only be a copy that can
    // go stale or die with the process. It is logged for the watcher, nothing
    // downstream depends on this line.
    log(`${piece.id} round ${round}: ${String(verdict.largest_gap).slice(0, 120)}`)
  }
  return { ...last, stopped: 'round cap reached' }
}

phase('Build')

// The dependency graph, expressed exactly and no more tightly than it is real.
// P10, P11 and P12 depend only on wave-one pieces that have already closed, so
// they are mutually independent and run concurrently. P13 is different: its
// frontend path src/plan/break/media/** sits INSIDE P10 src/plan/break/**, so
// running them together is a filesystem collision, not a scheduling preference.
// Chaining P13 behind P10 inside one thunk makes it start the moment P10
// returns, rather than after the slowest of the other three, which is what a
// flat barrier between two build stages would have cost.
const results = await parallel([
  async () => {
    const p10 = await runPiece(PIECES.P10)
    if (p10 && p10.stopped === 'builder returned nothing') {
      log('P13 skipped: it extends P10 and P10 produced nothing to extend')
      return [p10]
    }
    const p13 = await runPiece(PIECES.P13,
      '\nP10 has already finished and frozen the break surface you extend. Read what it built, in full, before you write anything.')
    return [p10, p13]
  },
  async () => [await runPiece(PIECES.P11)],
  async () => [await runPiece(PIECES.P12)],
])

const pieces = results.filter(Boolean).flat().filter(Boolean)
log(`build phase complete: ${pieces.length} piece records`)

// The barrier above is the point. This integration critic cannot begin until
// every piece has returned, which is the structural guarantee that was missing
// when a verification pass was started by hand while builders were still in
// flight and attributed their work in progress to the previous wave.
phase('Verify')
const integration = await agent(`You are the INTEGRATION CRITIC for wave two of the Meridian experience rebuild. Every piece has finished. ultrathink.
${LAWS}
YOU ARE READ-ONLY on the repository: you may not edit any file. You may run tests and servers, write scratch under /tmp, drive a browser, and read anything.
The four pieces reported this:
${pieces.map((p) => `- ${p.piece} ${p.name}: ${p.stopped ? 'STOPPED, ' + p.stopped : 'passed on round ' + p.round}. Critic evidence: ${String(p.verdict && p.verdict.evidence || 'none').slice(0, 600)}`).join('\n')}

DO ALL OF THIS AND REPORT WHAT YOU MEASURED, NOT WHAT YOU EXPECT:
1. THE SUITE. Run ~/.venvs/meridian/bin/python -m pytest tests/ -q from the repo root. Scope matters: a bare pytest at the root also collects the vendored Google Meridian library under meridian/, whose own suite carries hundreds of failures unrelated to this product. Report the exact counts for tests/ only.
2. OWNERSHIP. git diff --stat and git status. Every changed path must sit in some piece wave-two row in section 8.2 of the spec, or be a wave-one path a piece legitimately took by handover. Name every violation exactly.
3. THE SEAMS. The pieces were built alone and must compose. Verify in a browser: P10 pod renders inside the P3 break surface, P13 media verification renders inside the P10 pod, P12 adoption reads the P7 release artifacts, P11 pacing sits with the P4 client view. A piece that works alone and breaks its neighbour is the failure this phase exists to catch.
4. FABRICATION, the sharpest test. There is no delivery feed and no media feed. Walk every new screen and confirm that every number is either computed from a real file or stated as an honest unknown naming the missing input. One guessed delivered figure or invented codec is a wave-level failure.
5. THE TRAINING LINE. P12 is a training piece. Confirm no operator surface offers an act whose output lands under models/, and that adoption is company-only.
Report the single largest remaining gap across the whole wave.`,
  { label: 'integration', phase: 'Verify', schema: INTEGRATION })

return { pieces, integration }
