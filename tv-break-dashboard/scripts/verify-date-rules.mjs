// The date guard.
//
// A rule that is not tested is not a rule. A pacing card printed six ISO dates
// comma separated into a Hebrew interface, and every part of that was fixable at
// the call site, which is why it kept happening: seventeen other surfaces were
// formatting dates for themselves, each with its own answer, and the answers
// disagreed on the format, on whether a run collapsed and on whether a date was
// protected from the direction around it.
//
// The answer now lives in src/shell/dates.js, and this script fails when a
// surface answers it anywhere else.
//
// EXCEPTIONS are budgets, not pardons. Each entry names a file, a rule and a
// count, and the count may only go down. A new violation in an already-excepted
// file still fails, and a file that drops below its budget fails too so the
// number is lowered rather than left as headroom for a regression.
//
// Run: npm run test:dates

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = 'src';

// The one file allowed to decide what a date looks like.
const HOME = 'src/shell/dates.js';

const RULES = [
  {
    id: 'date-format-outside-home',
    // Every way a surface can ask the platform to render a date or a clock for
    // it.
    //
    // The last two alternatives exist because toLocaleString with no options is
    // the shape that slips past everything: three files in src/sources printed a
    // file's modified time that way and read as number formatting. Number
    // formatting cannot reach either of them, because a number takes
    // fraction-digit options and none of the date option keys, and because a
    // count is never held in a variable called date, parsed, when, stamp,
    // instant or moment.
    files: /\.jsx?$/,
    pattern: /toLocale(?:Date|Time)String\s*\(|Intl\.DateTimeFormat\s*\(|new Date\([^)]*\)\s*\.toLocaleString\s*\(|\.toLocaleString\s*\([^)]*\b(?:year|month|day|hour|minute|second|timeZone|dateStyle|timeStyle)\s*:|\b(?:date|parsed|when|stamp|instant|moment)\w*\s*\.toLocaleString\s*\(/g,
    message: 'a date or clock formatted outside shell/dates.js. Use formatDay, formatDayList, formatSpan, formatStamp or formatClock.',
  },
  {
    id: 'date-list-join',
    // A list of days joined by hand. This is the reported defect itself: the
    // join produces a flat list of ISO strings with no runs collapsed and a
    // separator that a reader cannot tell from a range joiner. formatDayList
    // does the grouping, the shape and the isolation in one call.
    //
    // The lookbehind spares weekdays: a list of weekday NAMES is not a list of
    // calendar days and has no runs to collapse.
    files: /\.jsx?$/,
    pattern: /\b[\w.$]*(?:[Dd]ates|(?<![Ww]eek)[Dd]ays)\s*\.join\s*\(/g,
    message: 'a list of days joined by hand. Use formatDayList from shell/dates.js, which collapses consecutive runs.',
  },
  {
    id: 'date-part-arithmetic',
    // Pulling a date apart by hand: the last disguise, and the one that produces
    // a format nobody else in the product uses. It is also where the timezone
    // bugs live, because getDate on a Date built from a bare YYYY-MM-DD returns
    // the day before for a reader west of the broadcast zone.
    files: /\.jsx?$/,
    pattern: /\.get(?:UTC)?(?:Date|Month|FullYear|Day)\s*\(/g,
    message: 'calendar parts read off a Date by hand. parseDay, formatDay and weekdayName in shell/dates.js do this once, on the string, with no zone to get wrong.',
  },
  {
    id: 'raw-iso-in-copy',
    // The quietest version of the defect, and the one the owner actually
    // reported: no formatting call at all, just a payload's ISO field dropped
    // straight into a sentence. It reads as a machine format because it is one.
    //
    // The field list is deliberately short and deliberately unambiguous: these
    // eight are calendar DAYS on this product's payloads, never weekday keys and
    // never instants. An interpolation that passes its value through anything
    // named format* is already going through the home and is not a hit.
    files: /\.jsx?$/,
    pattern: /\$\{(?:(?!format)[^}])*\b(?:date_from|date_to|starts_on|ends_on|expires_on|start_date|end_date|broadcast_date)\b[^}]*\}/g,
    message: 'a payload ISO date dropped into a string. Use formatDay, formatSpan or formatDayList from shell/dates.js so the reader sees dd/mm/yyyy.',
  },
];

// QUARANTINED DIRECTORIES.
//
// Held by live agents while this guard was written, so they could not be edited
// and their contents moved underneath it. A per-file count cannot be pinned
// against a directory somebody else is changing.
//
// This is a debt, not a decision. The violations inside are the same bug as
// everywhere else, and each is logged in docs/ux-gauntlet/state/sweep-backlog.jsonl
// with its file and line. Delete an entry here and sweep the directory the
// moment its holder lands, and the guard will then hold the line for it.
const QUARANTINED = [
  'src/clients/pacing/',
  'src/model/console/',
];

// Budgets. Each entry is a debt to pay down, with the reason it is still open. A
// place that genuinely needs its own arithmetic belongs here with the reason it
// does, not with the rule switched off.
const EXCEPTIONS = [
  // The month grid. monthMatrix walks a cursor day by day to build the weeks of
  // a calendar month including the adjacent-month padding, and localIsoDate
  // reads the cursor back. This is calendar CONSTRUCTION, not date rendering:
  // there is no reader-facing string here, only the ISO keys the grid is drawn
  // from, and every one of them is handed to shell/dates.js to be read out.
  { file: 'src/rules/calendar-events-lib.js', rule: 'date-part-arithmetic', count: 6 },
  // The same construction on the model surface: getDay maps a stored event's
  // date onto the planner's own Sun..Sat day KEY, which is a lookup and not a
  // reading.
  { file: 'src/rules/CalendarEventsModel.jsx', rule: 'date-part-arithmetic', count: 1 },

  // The four places an ISO date is CORRECT, because none of them is read by a
  // person. A key, a filename and two CSV cells are machine-facing, and
  // dd/mm/yyyy in any of them would be a bug: a spreadsheet parses the ISO form
  // unambiguously and cannot parse ours, and a filename has to sort.
  //
  // The dedupe key a paste-import builds from an event's name and start date.
  { file: 'src/rules/CalendarEvents.jsx', rule: 'raw-iso-in-copy', count: 1 },
  // The exported day-table's filename, which carries the window it covers.
  { file: 'src/today/TodayMoney.jsx', rule: 'raw-iso-in-copy', count: 2 },
  // The two scope cells at the head of that CSV, read by a spreadsheet.
  { file: 'src/today/today-export.js', rule: 'raw-iso-in-copy', count: 2 },
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
// discuss toLocaleDateString and getMonth: this file and dates.js both do it at
// length. Blank the comments before matching, preserving newlines so reported
// line numbers still point at the real line.
function withoutComments(source) {
  const blank = (m) => m.replace(/[^\n]/g, ' ');
  return source
    .replace(/\/\*[\s\S]*?\*\//g, blank)
    .replace(/(^|[^:])\/\/[^\n]*/g, (m, lead) => lead + blank(m.slice(lead.length)));
}

const files = walk(ROOT).map((f) => f.split('\\').join('/'));
const failures = [];
const stale = [];

for (const file of files) {
  for (const rule of RULES) {
    if (!rule.files.test(file)) continue;
    if (file === HOME) continue;
    if (QUARANTINED.some((prefix) => file.startsWith(prefix))) continue;
    const raw = readFileSync(file, 'utf8');
    const source = withoutComments(raw);
    const rawLines = raw.split('\n');
    const hits = [];
    source.split('\n').forEach((line, index) => {
      const found = line.match(new RegExp(rule.pattern.source, 'g'));
      if (found) hits.push({ line: index + 1, text: rawLines[index].trim().slice(0, 100), n: found.length });
    });
    const total = hits.reduce((sum, h) => sum + h.n, 0);
    const budget = budgetFor(file, rule.id);
    if (total > budget) failures.push({ file, rule, hits, total, budget });
    else if (budget > 0 && total < budget) stale.push({ file, rule: rule.id, total, budget });
  }
}

if (failures.length === 0 && stale.length === 0) {
  console.log('Date rules hold. A calendar day is read in src/shell/dates.js and nowhere else.');
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
console.error(`\nDate guard failed: ${totalOver} violations over budget in ${failures.length} files, ${stale.length} stale budgets.`);
process.exit(1);
