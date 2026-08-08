// The card guard.
//
// A rule that is not tested is not a rule. The owner sent screenshots of the
// Break Library and asked why, inside one card, the header and the prose sit
// inset from the card's edge while the rows run flush against the border.
// Measured in Hebrew on the shipped tree: the card title sat 17px from the
// card's edge and the first cell of the first row sat 1px from it, a 16px
// misalignment, and the ranked-breaks grid and the compliance ledger did the
// same thing. None of those surfaces was the cause. The cause was that a card
// owned no inset at all, so every child invented its own and a child that
// invented none ran to the border.
//
// The answer now lives in src/shell/card.css and src/shell/primitives.jsx, and
// this script fails when a surface answers it anywhere else.
//
// EXCEPTIONS are budgets, not pardons. Each entry names a file, a rule and a
// count, and the count may only go down. A new violation in an already-excepted
// file still fails, and a file that drops below its budget fails too so the
// number is lowered rather than left as headroom for a regression. This is the
// same ratchet as verify-direction-rules.mjs, deliberately.
//
// Run: npm run test:card

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = 'src';

// The two files that own the card. Nothing else may build one.
const CARD_SOURCE = 'src/shell/card.css';
const CARD_PRIMITIVE = 'src/shell/primitives.jsx';

// The spacing scale, from tokens.css. A padding written as one of these
// numbers in px is a spacing decision that already has a name, so it must use
// the name. Values NOT on this list are left alone on purpose: rounding 11px to
// 12px across a product without looking at what it does is how a sweep breaks
// things, and a 1px border or a 2px optically-tuned chip inset is not a
// spacing decision at all. See design-rules.md section 8.
const SCALE = {
  4: '--space-1',
  8: '--space-2',
  12: '--space-3',
  14: '--space-4',
  18: '--space-5',
  24: '--space-6',
};

const RULES = [
  {
    id: 'hand-built-card',
    // The recipe for a card: a surface, a hairline border and a radius. Written
    // out by hand it produces a box that LOOKS like a card and knows nothing
    // about the card's inset, which is precisely how a table came to run flush
    // to a border under an inset header. Use .card, or the Card primitive.
    files: /\.css$/,
    message: 'a card built by hand. Use the Card primitive from shell/primitives.jsx, or the .card class. The inset of everything inside a card lives in shell/card.css.',
  },
  {
    id: 'off-scale-padding',
    // A padding whose value is a scale step written as a number.
    files: /\.css$/,
    message: 'a padding written in px where a spacing token says the same thing. Use the token so a change to the scale reaches this rule.',
  },
];

// QUARANTINED DIRECTORIES.
//
// Held by other agents while this sweep ran, so they could not be edited and
// their contents moved underneath it. A per-file count cannot be pinned against
// a directory someone else is changing, so the whole prefix is excused until it
// is released.
//
// This is a debt, not a decision. The violations inside are the same bug as
// everywhere else, and each one is written down in
// docs/ux-gauntlet/state/sweep-backlog.jsonl with the fix. In particular
// src/plan/break/break-board.css holds `.break-table`, which is the SECOND of
// the two tables in the owner's own Break Library screenshot: measured after
// this sweep its cells still sit 9px from the card edge under a title at 17px.
// Delete an entry here and sweep the directory the moment its holder lands.
const QUARANTINED = [
  'src/clients/pacing/',
  'src/plan/break/',
  'src/model/console/',
  // Not frozen, but carrying another agent's uncommitted work at sweep time.
  'src/model/candidates/',
];

// Budgets. Each entry is a debt to pay down, with the reason it is still open.
const EXCEPTIONS = [
  // --- off-scale-padding: NOTHING. -----------------------------------------
  //
  // There is no budget for this rule and there should never be one. Every
  // padding value that a token already names was swept onto the token: 233 of
  // them, plus 51 restatements of the card inset. The substitution is
  // value-identical by construction, --space-3 IS 12px, and that was proved by
  // expanding every token back to its number and diffing against the commit:
  // 49 stylesheets came out byte-identical. So a violation of this rule is
  // always new, always mechanical, and always fixable on the spot.
  //
  // The one survivor in the whole tree is src/model/console/model-console.css
  // line 203, inside a quarantined directory, and it is in the backlog.

  // --- hand-built-card ------------------------------------------------------
  //
  // Fifty-four boxes that draw a surface, a hairline and a radius without being
  // the card primitive. Each is a real card and each is a real debt.
  //
  // They are budgeted rather than swept, and the reason is not squeamishness.
  // Converting one is a JSX change: the container has to become <Card>, and its
  // children have to be sorted into the ones that take the inset and the one
  // that deliberately bleeds. That is a judgement about what a surface is FOR,
  // which is the definition of work that belongs to the piece that owns it
  // rather than to a sweep. Doing it silently from the stylesheet would move
  // the layout of pages nobody measured, and this campaign has already paid for
  // that lesson more than once.
  //
  // What the budget buys is the thing that actually holds: no fifty-fifth. The
  // count may only go down, and every entry is written out in
  // docs/ux-gauntlet/state/sweep-backlog.jsonl with its file, its line and the
  // class, so the surface that owns it can close it without re-deriving it.
  { file: 'src/clients/clients-campaigns.css', rule: 'hand-built-card', count: 1 },
  { file: 'src/clients/clients-delivery.css', rule: 'hand-built-card', count: 1 },
  { file: 'src/clients/clients-record.css', rule: 'hand-built-card', count: 1 },
  { file: 'src/clients/clients-tree.css', rule: 'hand-built-card', count: 1 },
  { file: 'src/clients/clients-workspace.css', rule: 'hand-built-card', count: 2 },
  { file: 'src/history/history-detail.css', rule: 'hand-built-card', count: 1 },
  { file: 'src/history/history.css', rule: 'hand-built-card', count: 2 },
  { file: 'src/kai/assistant-console.css', rule: 'hand-built-card', count: 8 },
  { file: 'src/plan/day/day-board.css', rule: 'hand-built-card', count: 2 },
  { file: 'src/plan/day/day-readout.css', rule: 'hand-built-card', count: 2 },
  { file: 'src/plan/day/override-console.css', rule: 'hand-built-card', count: 1 },
  { file: 'src/plan/week/plan-week-goal.css', rule: 'hand-built-card', count: 1 },
  { file: 'src/plan/week/plan-week-palette.css', rule: 'hand-built-card', count: 1 },
  { file: 'src/plan/week/plan-week-panels.css', rule: 'hand-built-card', count: 1 },
  { file: 'src/plan/week/plan-week.css', rule: 'hand-built-card', count: 2 },
  { file: 'src/rules/calendar-month-grid.css', rule: 'hand-built-card', count: 1 },
  { file: 'src/rules/pricing-management.css', rule: 'hand-built-card', count: 1 },
  { file: 'src/rules/rules-restrictions.css', rule: 'hand-built-card', count: 2 },
  { file: 'src/rules/rules-workspace.css', rule: 'hand-built-card', count: 1 },
  // The sign-in screen is rendered before a locale, a shell or a card exists,
  // which is why it carries its own surface. It is the one entry here with an
  // argument for staying rather than a plan for leaving.
  { file: 'src/shell/login.css', rule: 'hand-built-card', count: 2 },
  // The 6,000-line shell sheet, and the largest single debt. It was 18 before
  // this sweep; .page-panel and .analytics-panel moved out of it into card.css
  // and became the primitive, which is where the two came off.
  { file: 'src/shell/styles.css', rule: 'hand-built-card', count: 16 },
  { file: 'src/sources/sources-findings.css', rule: 'hand-built-card', count: 1 },
  { file: 'src/sources/sources.css', rule: 'hand-built-card', count: 2 },
  { file: 'src/today/today-controls.css', rule: 'hand-built-card', count: 1 },
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
// discuss padding: 16px and border-radius: this file and card.css both do it at
// length. Blank the comments before matching, preserving newlines so reported
// line numbers still point at the real line.
function withoutComments(source) {
  const blank = (m) => m.replace(/[^\n]/g, ' ');
  return source.replace(/\/\*[\s\S]*?\*\//g, blank);
}

// The selector and the declarations of a CSS rule sit on different lines, and
// whether a block is a card depends on reading its whole body at once. Split
// the sheet into blocks and carry the selector with the body.
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

// Is this block a card, or is it a control that happens to share the recipe?
//
// A card is a BLOCK that holds other things. A button, an input, a chip, a tab
// and a badge all draw a surface, a hairline and a radius too, and calling one
// of those a card is how a guard earns its reputation for crying wolf. They are
// told apart on three signals, all of them present in the block itself.
function isHandBuiltCard(selector, body) {
  const hasSurface = /background:[^;]*var\(--surface\)/.test(body);
  const hasHairline = /(?:^|[\s;])border:\s*1px solid/.test(body);
  const hasRadius = /border-radius:\s*[^;]+;/.test(body);
  if (!(hasSurface && hasHairline && hasRadius)) return false;

  // A pill is a control, never a card.
  if (/border-radius:\s*var\(--radius-pill\)/.test(body)) return false;
  // Inline layout is a control, never a card.
  if (/display:\s*inline-(flex|block|grid)/.test(body)) return false;
  // A form control or a run of code, by element or by name.
  if (/\b(button|input|select|textarea|code|summary|label)\b/.test(selector)) return false;
  if (/(chip|badge|tab|pill|toggle|btn|-act\b|-close\b|search|filter|swatch|dot)/i.test(selector)) return false;
  // A control that names itself by what it does rather than what it is.
  if (/cursor:\s*pointer/.test(body) && !/overflow:/.test(body)) return false;
  return true;
}

const files = walk(ROOT).map((f) => f.split('\\').join('/'));
const failures = [];
const stale = [];

for (const file of files) {
  for (const rule of RULES) {
    if (!rule.files.test(file)) continue;
    if (file === CARD_SOURCE || file === CARD_PRIMITIVE) continue;
    if (QUARANTINED.some((prefix) => file.startsWith(prefix))) continue;
    const raw = readFileSync(file, 'utf8');
    const source = withoutComments(raw);
    const rawLines = raw.split('\n');
    const hits = [];

    if (rule.id === 'hand-built-card') {
      for (const block of cssBlocks(source)) {
        if (isHandBuiltCard(block.selector, block.body)) {
          hits.push({ line: block.line, text: `${block.selector.replace(/\s+/g, ' ')} { ... }`.slice(0, 100), n: 1 });
        }
      }
    } else {
      source.split('\n').forEach((line, index) => {
        const match = line.match(/(?:^|[\s;{])padding(?:-[a-z-]+)?\s*:\s*([^;}]+)/);
        if (!match) return;
        const found = (match[1].match(/\b(\d+)px/g) || [])
          .map((v) => Number(v.replace('px', '')))
          .filter((v) => SCALE[v]);
        if (found.length) {
          const names = found.map((v) => `${v}px -> var(${SCALE[v]})`).join(', ');
          hits.push({ line: index + 1, text: `${rawLines[index].trim().slice(0, 60)}   [${names}]`, n: found.length });
        }
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
  console.log('Card rules hold. The card and its inset live in src/shell/card.css and nowhere else.');
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
console.error(`\nCard guard failed: ${totalOver} violations over budget in ${failures.length} files, ${stale.length} stale budgets.`);
process.exit(1);
