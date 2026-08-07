// The direction guard.
//
// A rule that is not tested is not a rule. The dashboard is bilingual and the
// Hebrew side is right-to-left, so every surface that prints a number, a date or
// a Latin identifier has to say how that run relates to the direction around it.
// Five hundred sites each answered that question for themselves, and the answers
// disagreed: the shipped week-compare table put its numeric cells 321px, 320px
// and 212px from the cell's right edge while the row header sat at 8px, because
// a dir attribute on a cell re-anchors what text-align: start resolves against.
//
// The answer now lives in src/shell/bidi.jsx and in three CSS rules, and this
// script fails when a surface answers it anywhere else.
//
// EXCEPTIONS are budgets, not pardons. Each entry names a file, a rule and a
// count, and the count may only go down. A new violation in an already-excepted
// file still fails, and a file that drops below its budget fails too so the
// number is lowered rather than left as headroom for a regression.
//
// Run: npm run test:direction

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const ROOT = 'src';

// The one file allowed to state direction for the whole dashboard.
const PRIMITIVE_SOURCE = 'src/shell/bidi.jsx';
// The one stylesheet block allowed to carry the primitive's own rules. The guard
// checks the class names rather than the line numbers so the rules can move.
const PRIMITIVE_CLASSES = ['.bidi-figure', '.bidi-code', '.bidi-name'];

const RULES = [
  {
    id: 'jsx-dir',
    // A dir attribute is a direction OVERRIDE: it fixes the run's internal order
    // and re-anchors the element's alignment. Isolation does the first without
    // the second, and that is what the primitive gives you.
    files: /\.jsx?$/,
    pattern: /\bdir=(?:"[^"]*"|\{)/g,
    message: 'dir attribute. Use Figure, Code or Name from shell/bidi.jsx, which isolates without re-anchoring alignment.',
  },
  {
    id: 'css-physical-align',
    // text-align: start/end already mean "the reading edge" in both locales.
    files: /\.css$/,
    pattern: /text-align:\s*(left|right)\b/g,
    message: 'physical text-align. Use start or end so the column follows the document direction.',
  },
  {
    id: 'css-physical-box',
    // Every one of these has a logical twin that flips with the direction.
    files: /\.css$/,
    pattern: /(?:^|[\s;{])(margin|padding|border)-(left|right)\s*:/g,
    message: 'physical box property. Use the -inline-start / -inline-end form.',
  },
  {
    id: 'css-physical-offset',
    // left/right as positioning offsets; inset-inline-start/end flip correctly.
    files: /\.css$/,
    pattern: /(?:^|[\s;{])(left|right)\s*:\s*(?!auto\b)/g,
    message: 'physical offset. Use inset-inline-start / inset-inline-end.',
  },
  {
    id: 'css-direction',
    // direction and unicode-bidi belong to the primitive's three rules only.
    files: /\.css$/,
    pattern: /(?:^|[\s;{])(direction|unicode-bidi)\s*:/g,
    message: 'direction or unicode-bidi outside the primitive. Add the class from shell/bidi.jsx to the element instead.',
  },
  {
    id: 'literal-bidi-mark',
    // The same rule again, in its last disguise: the isolate characters typed
    // straight into a template string. They render as nothing, so a stray or
    // unpaired one is invisible in review, and each site that types them is
    // choosing an isolate kind for itself. The primitive's isolate function
    // picks one kind, first-strong, and is the only thing that should hold
    // these characters. Its own source writes them as escapes for this reason.
    files: /\.jsx?$/,
    pattern: /[\u2066-\u2069\u200e\u200f]/g,
    message: 'literal bidi control character in a string. Use isolate() from shell/bidi.jsx, or the Figure, Code and Name components where an element can go.',
  },
];

// QUARANTINED DIRECTORIES.
//
// These three were held by other agents while this sweep ran, so they could not
// be edited and their contents moved underneath it: PacingRow.jsx went from two
// violations to four in the time the guard was being written. A per-file count
// cannot be pinned against a directory someone else is changing, so the whole
// prefix is excused until it is released.
//
// This is a debt, not a decision. The violations inside are the same bug as
// everywhere else. PacingDays.jsx line 89 is the td dir="ltr" behind the column
// the owner reported. Delete an entry here and sweep the directory the moment
// its holder lands, and the guard will then hold the line for it.
const QUARANTINED = [
  'src/clients/pacing/',
  'src/plan/break/',
  'src/model/console/',
];

// Budgets. Each entry is a debt to pay down, with the reason it is still open.
// A place that genuinely needs the physical form belongs here with the reason it
// does, not with the rule switched off.
const EXCEPTIONS = [
  // Login and account fields: username, password and one-time code. dir on an
  // <input> sets the order digits and Latin characters are INSERTED in as the
  // operator types, which is not alignment and has no element to wrap. The
  // companion rule in login.css keeps those fields aligned to the reading edge.
  { file: 'src/shell/Login.jsx', rule: 'jsx-dir', count: 5 },
  { file: 'src/shell/UserAdminDialog.jsx', rule: 'jsx-dir', count: 3 },
  { file: 'src/shell/login.css', rule: 'css-direction', count: 1 },
  // .toast centres itself with left: 50% and translateX(-50%). This pair is
  // direction-neutral by construction: inset-inline-start would flip in Hebrew
  // and the translate would then push the toast off centre. Physical is correct.
  { file: 'src/shell/styles.css', rule: 'css-physical-offset', count: 1 },
  // Three direction statements survive in styles.css.
  //
  // .chart-ltr is the charting contract: an axis runs low to high left to right
  // in both locales, so a chart frame is a direction root of its own.
  //
  // NOT RESOLVED, and left alone rather than guessed at:
  // .brand-lockup inside the narrow-screen media query forces the logo lockup
  // left-to-right. Whether the mark belongs on the left in a Hebrew shell is a
  // branding decision, not a mechanics one, and it needs the owner.
  // .schedule-printable inside @media print restates rtl for the print region.
  // It reads as redundant, since the selector is already under .kairos-shell.rtl
  // which inherits, but print rendering was not verified here so it stands.
  { file: 'src/shell/styles.css', rule: 'css-direction', count: 3 },

  // .day-hours is a bar chart: each bar is an hour of the broadcast day laid out
  // left-to-right from midnight regardless of locale. The direction is content
  // semantics, not alignment, and flipping it would reverse the time axis.
  { file: 'src/plan/day/DayBoardReadout.jsx', rule: 'jsx-dir', count: 1 },
  // Two <input> fields: one for a clock string (HH:MM:SS) and one for a duration
  // in seconds. Both accept digits and colons in LTR order; dir="ltr" controls
  // cursor insertion order inside the field, which is not alignment.
  { file: 'src/plan/day/DayBoardToolbar.jsx', rule: 'jsx-dir', count: 2 },
  // <input type="number"> for a break count. dir="ltr" controls digit insertion
  // order inside the numeric field.
  { file: 'src/plan/day/OverrideDecisions.jsx', rule: 'jsx-dir', count: 1 },
  // <input type="number"> for a break count. Same as above.
  { file: 'src/plan/day/ScheduleInspector.jsx', rule: 'jsx-dir', count: 1 },
  // .timeline-scroll is the horizontal scrolling track container. It lays
  // programme bands and breaks left-to-right by clock time contract; flipping it
  // to RTL would reverse the time axis so later times appeared on the left.
  { file: 'src/plan/day/schedule-track-view.jsx', rule: 'jsx-dir', count: 1 },
  // frontier-chart-frame carries the chart-ltr contract: the SVG Pareto curve
  // plots retention (X) and revenue (Y) on axes that run low-to-high left-to-right
  // regardless of locale. Flipping the container would mirror the curve and reverse
  // the X axis so higher retention appeared on the left.
  { file: 'src/plan/week/FrontierPanel.jsx', rule: 'jsx-dir', count: 1 },
  // Two chart containers: frontier-chart-frame (same as FrontierPanel above) and
  // frontier-axis-legend, which labels the X and Y axes and must stay aligned with
  // the LTR chart space so X label sits under the X axis and Y label under the Y
  // axis. An RTL legend would mirror the labels off their axes.
  { file: 'src/plan/week/FrontierScopeChart.jsx', rule: 'jsx-dir', count: 2 },
  // .bar-list with class chart-ltr: a horizontal bar chart plotting booked value
  // or minutes per hour of the broadcast day, laid out left-to-right by clock time
  // contract. Flipping it would reverse the hour axis.
  { file: 'src/plan/week/SupplyPanel.jsx', rule: 'jsx-dir', count: 1 },
  // <input type="date"> for the "since" date picker. dir="ltr" controls the
  // digit and separator order inside the native date field; the bidi primitive
  // renders a wrapping span and cannot be applied to an input element.
  { file: 'src/history/HistorySince.jsx', rule: 'jsx-dir', count: 1 },
  // Two <input type="date"> for the from/until day window. Same reason as above.
  { file: 'src/history/HistoryReach.jsx', rule: 'jsx-dir', count: 2 },
  // .rows-drawer-table is the raw CSV preview: columns come from the source
  // file's own header row and may be Hebrew (daily file) or Latin (spots file).
  // The table grid must mirror the file's physical left-to-right column order
  // regardless of locale; flipping it would swap column positions from their
  // headers. A value inside a cell is not an exception: this covers only the
  // container.
  { file: 'src/sources/RowsDrawer.jsx', rule: 'jsx-dir', count: 1 },
  // Same as RowsDrawer above: the report preview table mirrors the CSV column
  // order from the source file and must stay physically left-to-right.
  { file: 'src/sources/ReportRowsDrawer.jsx', rule: 'jsx-dir', count: 1 },
  // .yield-bar-list.chart-ltr is a horizontal bar chart: each bar is a daypart
  // or programme type ranked by yield, laid out left-to-right with the largest
  // bar extending rightward. Flipping the container would reverse the bar axis
  // so a larger bar extended leftward, which is the wrong reading direction for
  // a magnitude scale.
  { file: 'src/today/YieldView.jsx', rule: 'jsx-dir', count: 1 },
  // <textarea> in the composer. The operator types bilingual questions; dir on a
  // textarea controls per-character cursor insertion order, which is content
  // semantics, not alignment. The bidi primitive renders a <span> and cannot be
  // applied to a textarea attribute.
  { file: 'src/kai/AssistantComposer.jsx', rule: 'jsx-dir', count: 1 },
  // <input> rename field for a conversation title that may be Hebrew or Latin.
  // Same reason as the textarea above: dir on an input controls cursor insertion
  // order inside the field. The bidi primitive cannot replace an input attribute.
  { file: 'src/kai/AssistantConversationsRail.jsx', rule: 'jsx-dir', count: 1 },
  // Two <p dir="auto"> in the model-text renderer (RichText and RetractedText).
  // Each paragraph needs its own direction from its first strong character because
  // the model writes in both Hebrew and English. Name is an inline <span> and
  // cannot replace a block-level <p>; dir="auto" is the only per-line mechanism.
  { file: 'src/kai/AssistantThread.jsx', rule: 'jsx-dir', count: 2 },
  // <Dialog> renders into a MUI portal outside the document root, so it cannot
  // inherit the root-level direction. dir must be stated explicitly on the Dialog
  // itself or the confirm panel renders as LTR inside an RTL page.
  { file: 'src/clients/AgencyDetailDrawer.jsx', rule: 'jsx-dir', count: 1 },
  // Three <input> elements: two type="date" for the flight window and one
  // type="number" for spot count. dir="ltr" on an input controls cursor insertion
  // order inside the field, not alignment. The bidi primitive renders a <span>
  // and cannot be applied to an input attribute.
  { file: 'src/clients/CampaignFlights.jsx', rule: 'jsx-dir', count: 3 },
  // <input> fields for campaign dates and percents inside the Field helper.
  // dir={ltr ? 'ltr' : 'auto'} controls cursor insertion order inside the field;
  // same reason as CampaignFlights above.
  { file: 'src/clients/CampaignTerms.jsx', rule: 'jsx-dir', count: 1 },
  // Two <input> elements: one type="number" for a premium multiplier and one
  // type="text" for an advertiser spelling. dir controls cursor insertion order
  // inside the native field and cannot be replaced by the bidi primitive.
  { file: 'src/clients/ClientRuleCard.jsx', rule: 'jsx-dir', count: 2 },
  // <input> field in the onboarding Field helper. dir={ltr ? 'ltr' : 'auto'}
  // controls cursor insertion order; same reason as CampaignTerms above.
  { file: 'src/clients/OnboardClientFlow.jsx', rule: 'jsx-dir', count: 1 },
  // Root container of the candidate board widget. The board may be rendered in
  // any host (a portal, an iframe-like mount, a test harness) that does not
  // inherit the document direction. dir={locale === 'en' ? 'ltr' : 'rtl'} on
  // the outermost div is the only way to establish the direction context for the
  // entire widget tree. The bidi primitive renders an inline span and cannot
  // replace a container-level direction root.
  { file: 'src/model/candidates/CandidateBoard.jsx', rule: 'jsx-dir', count: 1 },
];

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

function budgetFor(file, rule) {
  const hit = EXCEPTIONS.find((e) => e.file === file && e.rule === rule);
  return hit ? hit.count : 0;
}

// A guard that cries wolf gets switched off. Prose in a comment is allowed to
// discuss dir="ltr" and direction: this file and bidi.jsx both do it at length.
// Blank the comments before matching, preserving newlines so reported line
// numbers still point at the real line.
function withoutComments(source) {
  const blank = (m) => m.replace(/[^\n]/g, ' ');
  return source
    .replace(/\/\*[\s\S]*?\*\//g, blank)
    .replace(/(^|[^:])\/\/[^\n]*/g, (m, lead) => lead + blank(m.slice(lead.length)));
}

// The selector and the declarations of a CSS rule sit on different lines, so a
// line-by-line reader cannot tell whether a direction belongs to the primitive's
// own rule. Split the sheet into blocks and carry the selector with the body.
function cssBlocks(source) {
  const blocks = [];
  let selector = '';
  let body = '';
  let depth = 0;
  let line = 1;
  let bodyLine = 1;
  for (const ch of source) {
    if (ch === '\n') line += 1;
    if (ch === '{') {
      depth += 1;
      if (depth === 1) { bodyLine = line; body = ''; continue; }
    }
    if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        blocks.push({ selector: selector.trim(), body, line: bodyLine });
        selector = '';
        continue;
      }
    }
    if (depth === 0) selector += ch;
    else body += ch;
  }
  return blocks;
}

const files = walk(ROOT).map((f) => f.split('\\').join('/'));
const failures = [];
const stale = [];

for (const file of files) {
  for (const rule of RULES) {
    if (!rule.files.test(file)) continue;
    if (file === PRIMITIVE_SOURCE) continue;
    if (QUARANTINED.some((prefix) => file.startsWith(prefix))) continue;
    const raw = readFileSync(file, 'utf8');
    const source = withoutComments(raw);
    const rawLines = raw.split('\n');
    const hits = [];
    if (rule.id === 'css-direction') {
      // The primitive's own rules are the one place a stylesheet may say this.
      for (const block of cssBlocks(source)) {
        if (PRIMITIVE_CLASSES.some((c) => block.selector.includes(c))) continue;
        const found = block.body.match(new RegExp(rule.pattern.source, 'g'));
        if (found) hits.push({ line: block.line, text: `${block.selector} { ... }`.slice(0, 100), n: found.length });
      }
    } else {
      source.split('\n').forEach((line, index) => {
        const found = line.match(new RegExp(rule.pattern.source, 'g'));
        if (found) hits.push({ line: index + 1, text: rawLines[index].trim().slice(0, 100), n: found.length });
      });
    }
    const total = hits.reduce((sum, h) => sum + h.n, 0);
    const budget = budgetFor(file, rule.id);
    if (total > budget) {
      failures.push({ file, rule, hits, total, budget });
    } else if (budget > 0 && total < budget) {
      stale.push({ file, rule: rule.id, total, budget });
    }
  }
}

if (failures.length === 0 && stale.length === 0) {
  console.log('Direction rules hold. Isolation lives in src/shell/bidi.jsx and nowhere else.');
  process.exit(0);
}

for (const f of failures) {
  const over = f.total - f.budget;
  console.error(`\n${f.file}: ${f.total} x ${f.rule.id} (budget ${f.budget}, ${over} over)`);
  console.error(`  ${f.rule.message}`);
  for (const h of f.hits.slice(0, 6)) console.error(`  line ${h.line}: ${h.text}`);
  if (f.hits.length > 6) console.error(`  ... and ${f.hits.length - 6} more lines`);
}

for (const s of stale) {
  console.error(`\n${s.file}: ${s.rule} is down to ${s.total} but the budget still says ${s.budget}.`);
  console.error('  Lower the count in EXCEPTIONS so the budget cannot hide a regression.');
}

const totalOver = failures.reduce((sum, f) => sum + (f.total - f.budget), 0);
console.error(`\nDirection guard failed: ${totalOver} violations over budget in ${failures.length} files, ${stale.length} stale budgets.`);
process.exit(1);
