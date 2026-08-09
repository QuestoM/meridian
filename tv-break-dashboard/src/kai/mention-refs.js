// The offset algebra behind a chip, and nothing else.
//
// It is a plain module with no imports so a test can DRIVE it in node against
// the shipped code rather than describe it, which is the pattern mention-trigger
// and kai-keep-warm already set here.
//
// THE PROBLEM THIS SOLVES. A mention is a TYPED reference, {type, id}, and the
// composer is a plain textarea holding plain text. The design that avoids a
// contenteditable in a right-to-left dock is: the text carries the human-
// readable LABEL, and a side list carries {start, len, type, id, label} spans
// into it. The sentence therefore still reads as a Hebrew sentence, a copy-paste
// of it yields readable prose, and the typed identity rides alongside.
//
// The cost of that design is exactly this file: the spans have to survive the
// operator editing the text around them.
//
// WHAT IT DOES WITH AN EDIT, AND WHY IT DROPS RATHER THAN GUESSES.
//
// Any single edit -- a keystroke, a deletion, a paste, a cut, an autocomplete,
// a native undo -- shows up as one contiguous replaced region, which is
// recovered exactly by taking the common prefix and the common suffix of the old
// and new text. From there:
//
//   a span entirely BEFORE the edit is untouched;
//   a span entirely AFTER it shifts by the length delta;
//   a span that OVERLAPS the edit is DROPPED.
//
// The third is the whole judgement. An operator who edits the letters of an
// inserted name is changing what the sentence says, and a binding kept across
// that would send {type, id} for one object while the prose names another --
// the exact silent-drop failure this piece exists to prevent, inverted. Dropping
// is visible: the chip stops being a chip, the strip below the composer loses a
// row, and the words that remain are ordinary prose the free-text path still
// resolves. Nothing is lost that was not already ambiguous.
//
// A multi-region edit (a find-and-replace, a scripted change) is not
// representable this way, and the prefix/suffix recovery then spans everything
// between the two changes, so every reference inside that range drops. Erring
// toward dropping is the same judgement made once more.

// WHICH ARROW GOES IN, AND IT IS NEVER HARDCODED.
//
// The leading edge is the edge a line starts at, so in a right-to-left document
// it is the right and the key that moves INTO a row is ArrowLeft. It is the same
// gesture a file tree makes with ArrowRight in English, and getting it backwards
// would not be cosmetic: it would put the drill on the key that ascends and make
// the whole ladder unreachable in the language this product is written in.
//
// It takes the DIRECTION rather than the locale, so this module keeps its
// property of having no imports and being drivable in node, and so the one place
// that turns a locale into a direction stays shell/bidi.jsx's documentDirection.
export function edgeKeys(direction) {
  return direction === 'rtl'
    ? { descend: 'ArrowLeft', ascend: 'ArrowRight' }
    : { descend: 'ArrowRight', ascend: 'ArrowLeft' };
}

// The single contiguous region that differs between two strings, as
// {start, removed, inserted}. Identical strings give a zero-width region at the
// end, which shifts nothing and overlaps nothing.
export function editRegion(before, after) {
  const older = String(before || '');
  const newer = String(after || '');
  const max = Math.min(older.length, newer.length);
  let start = 0;
  while (start < max && older[start] === newer[start]) start += 1;
  let tail = 0;
  while (tail < max - start && older[older.length - 1 - tail] === newer[newer.length - 1 - tail]) tail += 1;
  return { start, removed: older.length - tail - start, inserted: newer.length - tail - start };
}

// Carry a list of spans across one text change.
export function shiftRefs(before, after, refs) {
  const list = Array.isArray(refs) ? refs : [];
  if (!list.length || before === after) return list;
  const { start, removed, inserted } = editRegion(before, after);
  const end = start + removed;
  const delta = inserted - removed;
  const out = [];
  for (const ref of list) {
    const from = ref.start;
    const to = ref.start + ref.len;
    if (to <= start) { out.push(ref); continue; }
    if (from >= end) { out.push({ ...ref, start: from + delta }); continue; }
    // Overlapped: the operator edited inside the name. The binding goes.
  }
  return out;
}

// Record the span an insertion just created, with the spans around it already
// carried across the same change. The picker hands the whole reference, so the
// type and the identifier are what the store said they were and nothing here
// invents either.
export function addRef(before, after, refs, span) {
  const carried = shiftRefs(before, after, refs);
  const next = [...carried, { ...span }];
  next.sort((a, b) => a.start - b.start);
  return next;
}

// Split the text into runs for the highlight overlay: plain stretches and the
// spans a reference covers, in reading order. The overlay paints the highlight
// behind the textarea's own glyphs, so what it must get right is the character
// count of every run and nothing else.
//
// A span whose recorded text no longer matches its label is not rendered as a
// chip. That cannot happen through shiftRefs, which drops an overlapped span
// outright, and it is checked anyway: this function is the last thing between a
// stale offset and a highlight painted over the wrong words.
export function chipRuns(text, refs) {
  const whole = String(text || '');
  const list = (Array.isArray(refs) ? refs : [])
    .filter((ref) => whole.slice(ref.start, ref.start + ref.len) === ref.label)
    .sort((a, b) => a.start - b.start);
  const runs = [];
  let cursor = 0;
  for (const ref of list) {
    if (ref.start < cursor) continue;
    if (ref.start > cursor) runs.push({ chip: false, text: whole.slice(cursor, ref.start) });
    runs.push({ chip: true, text: whole.slice(ref.start, ref.start + ref.len), type: ref.type, id: ref.id });
    cursor = ref.start + ref.len;
  }
  if (cursor < whole.length) runs.push({ chip: false, text: whole.slice(cursor) });
  return runs;
}

// The references still genuinely attached to this exact text. A span survives
// only while the characters it covers are still the label it was made from, so
// this is the one gate between the composer's state and what is sent.
export function liveRefs(text, refs) {
  const whole = String(text || '');
  return (Array.isArray(refs) ? refs : [])
    .filter((ref) => whole.slice(ref.start, ref.start + ref.len) === ref.label);
}
