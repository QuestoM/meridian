// What a refusal actually prints: the chip beside each finding, and which
// findings are worth a line at all. Pure functions with no React in them, so
// the rule the card renders by is the rule a test can measure directly.
//
// Two defects were measured on the shipped card and both are closed here.
//
// **The chip printed an internal token.** A finding owes a ``column`` key and a
// refusal is often about no column, so the door put ``<file>``, ``<header>`` or
// ``<frame>`` in that key and the card printed it verbatim, in bold Latin, on a
// Hebrew screen. The door now sends a ``scope`` code instead and the word is
// resolved here; a token that resolves to nothing prints nothing, which covers
// a report stored in the old shape and the frozen data contracts' own ``<frame>``.
//
// **The chip is a header the operator's own file carries.** The door used to
// send the LOADER's name for the column a finding is about, so a spots export
// whose dates will not parse printed a bold Latin `air_dt`, which is not in that
// file's header row and not in any export: it is a column the loader computes.
// The name is resolved against the candidate's own header row on the way out
// now, so what prints is `Date, Start time` there and `שעה` on the daily file,
// and the chip resolves its direction from the name rather than being forced.
//
// **The reason was printed twice.** The refusal panel prints the 400's detail
// and then the findings under it, and for a refusal made of one finding those
// are the same sentence, so a daily file missing one column said the same thing
// twice with a token between the two. A finding whose sentence is the one
// already printed above it adds nothing and is dropped, unless it names rows
// that sentence does not carry, in which case the rows print without it.
//
// The verdict over an accepted file is here for the same reason: it is a rule
// and not a layout, so a test runs it exactly as the card does.

import { SCOPE_LABELS, STATE_TONE, label } from './sources-copy.js';

// The tokens a finding can still arrive with in place of a column: from a
// report stored before the door sent a scope, and from the frozen data
// contracts, whose own frame-level violations are theirs to name.
const TOKEN_SCOPE = { '<file>': 'file', '<header>': 'header', '<frame>': 'frame' };

// The reason, in the language the rest of the card is in. The server writes
// every sentence it authors itself in both, and now writes a Hebrew sentence
// for the violations the frozen contracts raised too, from a count measured
// off the same rows rather than one parsed out of their frozen English, which
// stays unchanged in the other language. Hebrew falls back to the English
// sentence only for the rare finding this could not put a real count behind,
// never the other way.
export function findingMessage(finding, locale) {
  if (!finding) return '';
  if (locale === 'he') return String(finding.message_he || finding.message || '');
  return String(finding.message || '');
}

// Which scope this finding carries, or an empty string when it is about a
// column and not about the file as a whole.
export function findingScope(finding) {
  const column = String((finding && finding.column) || '');
  const code = String((finding && finding.scope) || '') || TOKEN_SCOPE[column] || '';
  return SCOPE_LABELS[code] ? code : '';
}

// The chip, or null when there is nothing true to put in it. A column name is
// the file's own header and resolves its own direction from its own first
// letter: the daily export's headers are Hebrew and the spots export's are
// Latin, and the door sends whichever one that file really carries. A scope is
// this product's word and reads in the card's own direction.
export function findingChip(finding, locale) {
  const scope = findingScope(finding);
  if (scope) return { text: label(SCOPE_LABELS, scope, locale), dir: 'auto' };
  const column = String((finding && finding.column) || '');
  if (!column || (column.startsWith('<') && column.endsWith('>'))) return null;
  return { text: column, dir: 'auto' };
}

// The heading each consequence code prints over a file the door accepted. The
// server answers for this candidate and not for its kind, so its own code
// decides, and a heading is added here the moment a code is added there.
const WARNED_HEADINGS = {
  replaces_live_input_with_no_rows: 'acceptedNoRows',
  replaces_live_input_with_warnings: 'acceptedWarned',
};

// The one effect that is not a loss. The door says what each warning cost the
// engine, and a value it could read two ways is not the same news as a field it
// never got: the first file is fully read, on one of the two readings, and the
// second reaches the engine with part of it missing. Anything else, including a
// warning the frozen data contracts raised and one from a door too old to say,
// is read as a loss, which is the safe way round.
const INTERPRETED = 'value_interpreted';

// The warnings the door let through rather than refusing, split by what they
// cost. An error is refused before the file can be committed, so on an accepted
// file this is the warn-severity half: the part that loads and reaches the
// engine anyway. Both halves are warnings and neither is a clean pass; they
// differ in what a person has to do next, which is why they do not share a
// heading.
export function warningEffects(findings) {
  const warned = (Array.isArray(findings) ? findings : []).filter((finding) => String((finding && finding.severity) || '') === 'warning');
  return {
    lost: warned.filter((finding) => String((finding && finding.effect) || '') !== INTERPRETED).length,
    interpreted: warned.filter((finding) => String((finding && finding.effect) || '') === INTERPRETED).length,
  };
}

// What the verdict panel over an accepted file says, and in which tone. Three
// outcomes an accepted file can carry are not good news, and each of them was
// printed teal under "the file passed every check" over an enabled commit
// button before it was closed: a file nothing will read, a file the engine will
// read that carries no rows, and a file the engine will read whose rows carry
// something it cannot read. The server's consequence code decides, and the
// findings on the same payload decide too, so neither one alone can put a green
// tick over a file the engine cannot fully read.
export function acceptedVerdict(check) {
  if (check && check.will_be_read === false) return { heading: 'acceptedNotRead', tone: 'warn' };
  const code = String(((check && check.consequence) || {}).code || '');
  if (code === 'replaces_live_input_with_no_rows') return { heading: WARNED_HEADINGS[code], tone: 'warn' };
  // Which warning it is decides the heading, because the consequence code says
  // only that there are warnings. A file whose only warning is that a date
  // could be read two ways is not a file that lost a field, and reading the
  // same heading over both is how the eleven days of a month whose day number
  // is twelve or under came to look like a broken export.
  const { lost, interpreted } = warningEffects(check && check.findings);
  const heading = lost ? 'acceptedWarned' : (interpreted && 'acceptedRead') || WARNED_HEADINGS[code] || '';
  return heading ? { heading, tone: 'warn' } : { heading: 'accepted', tone: 'ok' };
}

// The tone the state chip carries. The state itself is the server's word and it
// stays the server's word: a live file the engine reads that carries rows IS in
// use. What the chip may not do is read as a clean pass when the server's own
// remedy for that card says the last check of that same file came back with a
// warning. This is the third place the same rule lands, after the door's verdict
// and the card's remedy, and it is the one an eye reaches first.
export function stateTone(input) {
  const remedy = String(((input && input.remedy) || {}).code || '');
  if (remedy === 'in_use_with_warnings') return 'warn';
  return STATE_TONE[String((input && input.state) || 'missing')] || 'muted';
}

// The findings worth a line, each with the chip it prints and the sentence it
// prints. ``printed`` is the sentence already on screen above this list, and a
// finding that only restates it is not a second finding.
export function visibleFindings(findings, locale, printed) {
  const already = String(printed || '');
  const lines = [];
  for (const finding of Array.isArray(findings) ? findings : []) {
    const message = findingMessage(finding, locale);
    const repeats = Boolean(already) && message === already;
    const rows = Array.isArray(finding && finding.rows) ? finding.rows : [];
    if (repeats && rows.length === 0) continue;
    lines.push({ finding, chip: findingChip(finding, locale), message: repeats ? '' : message });
  }
  return lines;
}
