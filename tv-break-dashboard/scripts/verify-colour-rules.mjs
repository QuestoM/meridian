// A colour is written once, in tokens.css, and read everywhere else.
//
// WHY THIS EXISTS AS ITS OWN CHECK. `test:card` guards the SPACING half of the
// token row: a padding written as a bare 12px instead of var(--space-3) fails.
// Nothing guarded the COLOUR half. A raw #1a2035 or an rgba() dropped into a
// component stylesheet would have passed every check this repository had, which
// the wave-two sweep recorded as `unguarded-token-kinds` and nobody closed.
//
// I MEASURED THIS WRONG FIRST AND THE GUARD CORRECTED ME, which is the whole
// argument for writing the guard instead of running the grep. My grep reported
// ZERO literal colours outside tokens.css, and I believed it. The grep had a
// malformed flag, matched nothing, and printed 0. A measurement whose failure
// mode is a comfortable answer is not a measurement.
//
// The original count was 67, across eight stylesheets, most of them in the model
// console. So this is a RATCHET at the measured value rather than a clean-tree
// zero: it can only go down, and dropping BELOW the budget fails too, so the
// number here has to follow the tree down and cannot hide a regression in slack.
//
// It bans a literal colour, not a colour. `var(--teal)` is how you write one.
// tokens.css is the one file where the literal belongs, because that is what a
// token file IS.

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = 'src';
// The one file allowed to write a literal colour, and the reason it is allowed.
const TOKEN_SOURCE = 'src/tokens.css';

// Hex in 3, 4, 6 or 8 digits, and the functional notations. `currentColor`,
// `transparent` and `inherit` are keywords rather than values and stay legal:
// they say "the colour decided elsewhere", which is the rule rather than a
// breach of it.
const LITERAL_COLOUR = /(#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?|color-mix)\s*\()/g;

// Re-measured after the Studio Ledger token migration on 2026-08-15. Literal
// colours now live only in this token source, so there is no compatibility
// allowance to grow back.
const BUDGET = 0;

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) walk(path, out);
    else out.push(path);
  }
  return out;
}

const files = walk(ROOT)
  .map((f) => f.split('\\').join('/'))
  .filter((f) => f.endsWith('.css') && f !== TOKEN_SOURCE);

const found = [];
for (const file of files) {
  const lines = readFileSync(file, 'utf8').split('\n');
  // A comment explaining a colour is not a colour, and a comment can span lines.
  // Splitting on /* alone missed every continuation line, which is how a prose
  // sentence containing "rgb(17,24,39)" was counted as a breach on the first run.
  let inComment = false;
  lines.forEach((line, index) => {
    let code = '';
    let rest = line;
    while (rest.length) {
      if (inComment) {
        const close = rest.indexOf('*/');
        if (close === -1) { rest = ''; break; }
        inComment = false;
        rest = rest.slice(close + 2);
      } else {
        const open = rest.indexOf('/*');
        if (open === -1) { code += rest; break; }
        code += rest.slice(0, open);
        inComment = true;
        rest = rest.slice(open + 2);
      }
    }
    for (const match of code.matchAll(LITERAL_COLOUR)) {
      found.push(`${file}:${index + 1}  ${line.trim().slice(0, 90)}`);
    }
  });
}

if (found.length > BUDGET) {
  console.error(
    `A colour is written literally outside ${TOKEN_SOURCE}: ${found.length} against a budget of ${BUDGET}.`,
  );
  console.error('Add it to tokens.css and read it with var(--name), so one edit moves every surface.');
  for (const hit of found.slice(0, 20)) console.error(`  ${hit}`);
  process.exit(1);
}

if (found.length < BUDGET) {
  console.error(`Literal colours are down to ${found.length} but the budget still says ${BUDGET}.`);
  console.error('Lower BUDGET in this file so the slack cannot hide a regression.');
  process.exit(1);
}

console.log(`Colour rules hold. Every colour is written in ${TOKEN_SOURCE} and read everywhere else.`);
