// The @ trigger's grammar, and nothing else.
//
// It is a plain module with no imports so that a test can DRIVE it in node
// rather than grep the file that contains it, which is the pattern the keep-warm
// module already set here. What it decides is the one thing in this piece that
// can break something that used to work: whether a keystroke opens the picker.
// Getting that wrong does not produce a bad picker, it produces a composer that
// swallows keys while somebody is typing a sentence.
//
// So the grammar is deliberately narrow, and it is the rule both reference
// products converged on independently:
//
//   the @ must sit at a word boundary, so an address in mid-word never opens it;
//   the run after it may hold no whitespace, so a space closes it;
//   and it is read off the TEXT and the CARET, never off the keystroke, so a
//   paste, an undo and an arrow key all leave the state the text implies.
//
// The free-text path is untouched by all of it. An operator who never types @
// meets exactly the composer that shipped before this.

// The @ run the caret currently sits in, or null when there is none.
export function readMentionQuery(text, caret) {
  const upto = String(text || '').slice(0, caret);
  const at = upto.lastIndexOf('@');
  if (at < 0) return null;
  const before = at === 0 ? '' : upto[at - 1];
  if (before && !/\s/.test(before)) return null;
  const query = upto.slice(at + 1);
  if (/\s/.test(query)) return null;
  return { start: at, query };
}

// Replace the @query run with the object's own label and report where the caret
// belongs afterwards. One trailing space, because the operator is mid-sentence.
//
// What goes in is the STORE'S OWN NAME, as plain text. That is the whole R1
// contract with the model: the read tools already resolve that string, so a
// mention needs no new resolution path, and the same question typed by hand
// still reaches the same object.
// The WHOLE @ token goes, not just the part before the caret. A caret parked in
// the middle of a half-typed name is ordinary -- an operator backs up to fix a
// letter -- and replacing only the left half would leave the right half stranded
// after the inserted name. Measured by the node harness, which caught exactly
// that: `ask @co` with the caret after `c` produced `ask אסם o about it`.
export function insertMention(text, run, label) {
  const whole = String(text || '');
  const head = whole.slice(0, run.start);
  let end = run.start + 1 + run.query.length;
  while (end < whole.length && !/\s/.test(whole[end])) end += 1;
  const inserted = `${label} `;
  return { text: `${head}${inserted}${whole.slice(end)}`, caret: head.length + inserted.length };
}
