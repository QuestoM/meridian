// A link that cannot be seen, and the two ways this tree produced one.
//
// MEASURED on the Commercial page before this guard existed: 48 of 50 elements
// carrying the workspace link classes rendered with NO link signal at all — no
// underline, no link colour, no icon — and every one of the 48 was painted in
// --muted, the token this system reserves for DE-EMPHASIS. Client names,
// campaign titles and source-file links were all clickable and all looked like
// slightly faded prose. The report that started this was "nothing there is
// clickable", and it was literally true, including on hover.
//
// Neither cause is visible by reading the stylesheet, which is why they lived
// so long and why they are checked mechanically now.
//
// ONE. SPECIFICITY LOST TO A RUNTIME SHEET. MUI injects `.muirtl-XXXX { color;
// text-decoration: none }` into a <style> element it appends at runtime. That
// sheet lands AFTER every stylesheet in this repository, and one class against
// one class is equal specificity, so the later sheet wins outright — in the
// resting state and in :hover alike. Any rule that decides how a link LOOKS has
// to out-specify it. Doubling the class does that and costs nothing; an element
// selector would not, because these classes are worn by buttons in some places
// and anchors in others.
//
// TWO. AN UNDERLINE STYLED AND NEVER TURNED ON. text-decoration-color,
// -thickness, -style and text-underline-offset all do exactly nothing until
// text-decoration-line exists. The soft warm underline in
// studio-ledger-commercial.css was designed, reviewed and committed, and never
// rendered once.
//
// The budget only goes down. A count below it fails too, because slack is how a
// budget stops being a guard.

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Resolved against this file, not against the shell's cwd. A guard that only
// runs from one directory is a guard that quietly stops running.
const SRC = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'src');

// Measured 2026-08-19: zero of each, after the Commercial workspace was fixed.
const BUDGET_UNDERSPECIFIED = 0;
const BUDGET_DEAD_UNDERLINE = 0;

// Components whose className is beaten by the library's own runtime styles.
const THEMED = /<(?:Button|IconButton|Link|MenuItem|Chip|Tab|Pressable)\b[^>]*?className=["'{`]([^"'}`]+)/g;

const BLOCK = /([^{}]+)\{([^{}]*)\}/g;

// DECORATION ONLY, and only when it asks for one to EXIST.
//
// Whether the library sets a COLOUR depends on the variant a screen chose, so
// flagging colour reports a hundred filled buttons whose theme colour is
// exactly what they want. Whether it sets `text-decoration: none` does not
// depend on the variant — it is in the base style of every button it themes —
// so this is the half of the collision that can be decided by reading, and the
// half worth failing a build over: a text control's underline is the whole of
// its affordance. A rule declaring `none` and losing to a rule that also says
// none has lost nothing.
//
// Written as "capture the value, then test it" rather than as a lookahead. The
// lookahead version was wrong in a way worth keeping: `\s*` backtracks to zero
// width, the negative lookahead then succeeds against the SPACE before `none`,
// and every `text-decoration: none` in the tree reported as an underline. It
// rewrote nine filled buttons whose `none` agreed with the library perfectly.
const DECLARES_DECORATION = /text-decoration(?:-line)?\s*:\s*([^;]+)/g;
const PAINTS = (declarations) => {
  for (const [, value] of declarations.matchAll(DECLARES_DECORATION)) {
    if (value.trim() !== 'none') return true;
  }
  return false;
};
const DECORATION_PART = /text-decoration-(?:color|thickness|style)|text-underline-offset/;
const DECORATION_LINE = /text-decoration\s*:|text-decoration-line\s*:/;

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) walk(path, out);
    else out.push(path);
  }
  return out;
}

const files = walk(SRC);

// Which classes are actually worn by a themed component anywhere in the tree.
// Evidence, not a naming convention: a class called "-link" that never reaches
// MUI is none of this guard's business, and one called "amz-card-open" that
// does is very much its business.
const themedClasses = new Set();
for (const path of files.filter((f) => f.endsWith('.jsx') || f.endsWith('.js'))) {
  const text = readFileSync(path, 'utf8');
  for (const match of text.matchAll(THEMED)) {
    for (const name of match[1].split(/[\s${}?:'"`+]+/)) {
      if (/^[a-z][a-z0-9-]*$/.test(name)) themedClasses.add(name);
    }
  }
}

// Which selectors ever turn a decoration line on, so a hover rule that only
// recolours an underline the base rule established is not reported.
const turnsLineOn = new Set();
const sheets = files.filter((f) => f.endsWith('.css'));
for (const path of sheets) {
  const text = readFileSync(path, 'utf8');
  for (const [, selectors, declarations] of text.matchAll(BLOCK)) {
    if (!DECORATION_LINE.test(declarations)) continue;
    for (const one of selectors.split(',')) {
      turnsLineOn.add(one.trim().replace(/:[a-z-]+(\([^)]*\))?/g, ''));
    }
  }
}

const underspecified = [];
const deadUnderline = [];

for (const path of sheets) {
  const file = relative(SRC, path);
  const text = readFileSync(path, 'utf8');
  for (const match of text.matchAll(BLOCK)) {
    const [, selectors, declarations] = match;
    if (selectors.trim().startsWith('@')) continue;
    // The match begins immediately after the previous rule's brace, so the
    // selector's own line is found by skipping the whitespace between them.
    const lead = selectors.length - selectors.replace(/^\s+/, '').length;
    const line = text.slice(0, match.index + lead).split('\n').length;

    if (DECORATION_PART.test(declarations) && !DECORATION_LINE.test(declarations)) {
      const bare = selectors.split(',').map((s) => s.trim().replace(/:[a-z-]+(\([^)]*\))?/g, ''));
      if (!bare.some((s) => turnsLineOn.has(s))) {
        deadUnderline.push(`${file}:${line}  ${selectors.trim().slice(0, 70)}`);
      }
    }

    if (!PAINTS(declarations)) continue;
    for (const one of selectors.split(',')) {
      const selector = one.trim();
      // Exactly one class, optionally with a pseudo-class: specificity (0,1,0),
      // which a runtime sheet ties and then beats on order.
      const single = /^\.([a-z][a-z0-9-]*)(:[a-z-]+(\([^)]*\))?)?$/.exec(selector);
      if (single && themedClasses.has(single[1])) {
        underspecified.push(`${file}:${line}  ${selector}`);
      }
    }
  }
}

let failed = false;
const report = (found, budget, what, remedy) => {
  if (found.length > budget) {
    failed = true;
    console.error(`\n${what}: ${found.length}, budget ${budget}.`);
    found.slice(0, 24).forEach((f) => console.error(`  ${f}`));
    if (found.length > 24) console.error(`  ... and ${found.length - 24} more`);
    console.error(`  ${remedy}`);
  } else if (found.length < budget) {
    failed = true;
    console.error(
      `\n${what}: ${found.length}, below the budget of ${budget}. ` +
      `Lower the budget in this file, in the same commit that earned it.`);
  }
};

report(
  underspecified, BUDGET_UNDERSPECIFIED,
  'Link styles a runtime sheet will beat',
  'Write the class twice — .name.name — so the design system decides, not the component library.',
);
report(
  deadUnderline, BUDGET_DEAD_UNDERLINE,
  'Underlines styled but never turned on',
  'Add text-decoration-line, or drop the parts that do nothing without it.',
);

if (failed) process.exit(1);
console.log(
  'Link rules hold. Every themed link out-specifies the runtime sheet, ' +
  'and no underline is styled without being turned on.');
