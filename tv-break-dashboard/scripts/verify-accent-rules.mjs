// The one-sided accent bar, banned by the owner and now counted.
//
// design-rules.md section 1: a rule drawn down one edge of a box to mark it as
// important or as being in a state is banned. It reads as an unfinished frame,
// it inverts under right-to-left, and it MULTIPLIES: one appears, the next
// surface copies it, and there are twenty-six in seventeen files. That is
// measured, not hypothetical, which is why counting it is the whole point.
//
// The correct pattern already existed in shell/styles.css and was simply never
// applied anywhere else:
//
//     border: 1px solid var(--amber);
//     background: var(--amber-soft);
//     color: var(--amber);
//
// TWO ONE-SIDED DECLARATIONS ARE LEGITIMATE and this guard lets both through.
//
//   A STRUCTURAL DIVIDER at 1px solid var(--line). That is a rule BETWEEN two
//   things rather than an accent ON one thing.
//   A TRANSPARENT inline-start that reserves space for a marker which appears on
//   selection. That is layout, not decoration.
//
// The budget only goes down. A count below it fails too, because slack is how a
// budget stops being a guard, and this is the fourth ratchet in this repository
// to earn that sentence.

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const SRC = 'src';

// Measured 2026-08-09 across 11 files. Lower it in the same commit that removes
// bars; do not raise it.
const BUDGET = 0;

// A named exception needs a reason that would survive somebody asking why this
// one is different. There are none yet, deliberately: the owner asked for none
// anywhere, so the first entry here should be argued rather than assumed.
const EXCEPTIONS = [];

const ONE_SIDED = /border-inline-(?:start|end)(?:-width|-color)?\s*:\s*([^;]+);/;
const ALLOWED = [/var\(--line/, /transparent/, /^0$/, /^none$/];

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) walk(path, out);
    else if (path.endsWith('.css')) out.push(path);
  }
  return out;
}

const found = [];
for (const path of walk(SRC)) {
  const file = relative(SRC, path);
  if (EXCEPTIONS.includes(file)) continue;
  readFileSync(path, 'utf8').split('\n').forEach((line, index) => {
    const match = ONE_SIDED.exec(line);
    if (!match) return;
    const value = match[1].trim();
    if (ALLOWED.some((allowed) => allowed.test(value))) return;
    found.push({ file, line: index + 1, text: line.trim().slice(0, 96) });
  });
}

if (found.length > BUDGET) {
  console.error(`Accent bars went UP: ${found.length} against a budget of ${BUDGET}.`);
  console.error('design-rules.md section 1. Use a full border in the state colour with the');
  console.error('matching soft background. A one-sided rule is a divider only at var(--line).\n');
  for (const hit of found) console.error(`  ${hit.file}:${hit.line}: ${hit.text}`);
  process.exit(1);
}

if (found.length < BUDGET) {
  console.error(`Accent bars are down to ${found.length} but the budget still says ${BUDGET}.`);
  console.error('Lower BUDGET in this file so the slack cannot hide a regression.');
  process.exit(1);
}

console.log(`Accent bars at ${found.length}, and the budget only goes down.`);
