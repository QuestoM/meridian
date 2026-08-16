// Which rows the list shows: the search over the loaded page, and the folding
// that keeps a long feed readable at a glance.
//
// The request recorder records every mutating verb, and a large share of them
// save nothing: dragging a break scores the placement on every drop, the rate
// card prices a change before it is saved, Mabat reads the page on every dock
// open. Measured on the running instance, 57 of the 345 recorded requests in
// the newest 500 entries were one endpoint scoring placements nobody had saved.
//
// Those rows are real and they are not hidden. Adjacent previews of the same
// act by the same person on the same broadcast day become one row that carries
// the count and the window, and the detail lists every member. Nothing is
// dropped, nothing is summarised into a number the reader cannot open, and a
// change is never folded: an act that wrote to the operating record always gets
// its own line.

import { actorLabel, isoDay } from './history-labels';
import { actLabel, outcomeOf } from './history-refused';

// The search reads the row a person can see rather than the payload: the actor
// as it is printed, the act in the reader's own language, and the facts the row
// carries. It runs over the loaded page, which is what the footer says beside it
// whenever the page holds only part of the matched set.
//
// The act is read exactly as the row prints it, so a reader looking for what was
// refused types the word they can see and finds it.
export function matchesSearch(entry, needle, locale) {
  if (!needle) return true;
  const facts = entry.facts || {};
  const action = actLabel(facts.action, outcomeOf(entry), locale);
  const haystack = [
    entry.actor, actorLabel(entry.actor, locale), entry.kind, facts.path, facts.label,
    facts.source, facts.channel, facts.day, facts.run_id, facts.version_id, action,
    (facts.files || []).join(' '),
  ].filter(Boolean).join(' ').toLowerCase();
  return haystack.includes(needle);
}

// Two rows fold only when they are the same news. An act the server carried out
// and the same act refused are not, so the outcome joins the act in the key: a
// folded row prints one title and one of those two titles would be wrong.
function sameFold(a, b) {
  return a.kind === 'preview'
    && b.kind === 'preview'
    && a.actor === b.actor
    && (a.facts || {}).action === (b.facts || {}).action
    && outcomeOf(a) === outcomeOf(b)
    && isoDay(a.ts) === isoDay(b.ts);
}

// Entries arrive newest first, so the first member of a group is the newest and
// the last is the oldest. The folded row keeps the newest member's id, which is
// what an address in the URL points at.
export function foldPreviews(entries) {
  const out = [];
  entries.forEach((entry) => {
    const last = out.length ? out[out.length - 1] : null;
    if (last && sameFold(last, entry)) {
      last.members.push(entry);
      last.oldestTs = entry.ts;
      return;
    }
    out.push({ ...entry, members: [entry], oldestTs: entry.ts });
  });
  return out;
}

export function foldSize(entry) {
  return entry && Array.isArray(entry.members) ? entry.members.length : 1;
}
