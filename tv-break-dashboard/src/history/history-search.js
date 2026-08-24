// What the search actually covers, and why a list with no day window came back
// empty.
//
// The search runs in the browser over the entries the page loaded, and the page
// is a small window on the record. Measured on the running instance on
// 2026-08-02 at 02:4x: the record held 5,740 entries, the default page served
// 200 of them spanning twelve minutes, and typing `yieldo` into the box emptied
// the list under the sentence "nothing here matches those filters". The record
// held 18 entries by that operator at that moment: `GET
// /api/history?limit=200&actor=yieldo` answered matched 18, served 18. `yieldo`
// was in the payload's own `actors` list, so the dropdown two centimetres from
// the box found all 18 server side while the box beside it found none, and
// nothing on the row of controls said the two differ.
//
// The rule applied here is the one `emptyWindow` already applies under a day
// window, and it is this destination's oldest: name the control that emptied the
// list, print in the payload's own figures what dropping it would reveal, and
// carry every control that reaches the rest. Two things are added, because the
// search is the one control whose reach is smaller than the record's.
//
// 1. How much of the record the search actually looked at, as a figure. "It
//    holds 200 of 5,740, 3.5 percent of them" is the difference between a reader
//    concluding that nothing is there and a reader knowing where to look.
// 2. The operator filter, offered by name when the needle is an actor on the
//    record, because that is the control that answers and it answers over the
//    whole record rather than over this page.
//
// Nothing here counts anything: total, matched, served, newer, older, counts and
// actors are all the endpoint's own. This is plain JavaScript so the sentences
// can be executed by a test rather than grepped, which is the pattern
// history-runs.js established on this destination.

// The extension is explicit because this module is executed by node in a test,
// which is the point of it being plain JavaScript, and node resolves no other way.
import { KIND_LABELS, actorLabel } from './history-labels.js';

function count(value, locale) {
  return Number(value || 0).toLocaleString(locale === 'he' ? 'he-IL' : 'en-GB');
}

// How much of the set behind it this page holds, from the endpoint's own two
// figures and nothing else. A share that rounds away is stated as a bound rather
// than as 0.0, because 0.0 percent of 5,740 reads as none of it.
function share(served, matched, locale) {
  const pct = matched ? (Number(served) / Number(matched)) * 100 : 0;
  if (pct > 0 && pct < 0.1) return locale === 'he' ? 'פחות מ-0.1' : 'less than 0.1';
  return pct.toLocaleString(locale === 'he' ? 'he-IL' : 'en-GB', { maximumFractionDigits: 1 });
}

// The record itself is empty. The one sentence here that says so, and it may
// only be said when the endpoint counted nothing at all.
export const RECORD_EMPTY = [
  'Nothing has been recorded yet. A point is saved before every change and before every restore, so the first save will appear here.',
  'עדיין לא נרשם דבר. נקודה נשמרת לפני כל שינוי ולפני כל שחזור, כך שהשמירה הראשונה תופיע כאן.',
];

// The needle emptied the list. The sentence names the control rather than "the
// filters", because the search and the dropdown beside it do not reach the same
// distance and a reader has to know which one just answered.
export const SEARCH_MISS = [
  'Nothing on this page matches that search.',
  'שום דבר בעמוד הזה לא תואם את החיפוש.',
];

// What the search looked at, in the payload's own figures. The second clause is
// the whole point: a page is not the record, and this says by how much.
export function searchCoversLine(body, filtered) {
  const source = body || {};
  const served = [count(source.served, 'en'), count(source.served, 'he')];
  const matched = [count(source.matched, 'en'), count(source.matched, 'he')];
  const part = [share(source.served, source.matched, 'en'), share(source.served, source.matched, 'he')];
  if (filtered) {
    return [
      `The search covers this page, which holds ${served[0]} of the ${matched[0]} entries the other filters match, ${part[0]} percent of them.`,
      `החיפוש פועל על העמוד הזה, שמחזיק ${served[1]} מתוך ${matched[1]} הרשומות שתואמות את שאר הסינון, ${part[1]} אחוז מהן.`,
    ];
  }
  return [
    `The search covers this page, which holds ${served[0]} of the ${matched[0]} entries on the record, ${part[0]} percent of them.`,
    `החיפוש פועל על העמוד הזה, שמחזיק ${served[1]} מתוך ${matched[1]} הרשומות שברישום, ${part[1]} אחוז מהן.`,
  ];
}

// The needle read as an actor on the record. Exact first, in the stored name and
// in both of its printed forms, and otherwise the single actor it is contained
// in: one candidate is an answer, two are a guess and this offers none.
export function actorForNeedle(actors, needle) {
  const text = String(needle || '').trim().toLowerCase();
  if (!text) return '';
  const names = (actors || []).map((name) => String(name)).filter(Boolean);
  const forms = (name) => [name, actorLabel(name, 'en'), actorLabel(name, 'he')].map((form) => form.toLowerCase());
  const exact = names.find((name) => forms(name).includes(text));
  if (exact) return exact;
  const near = names.filter((name) => forms(name).some((form) => form.includes(text)));
  return near.length === 1 ? near[0] : '';
}

// Why that filter is the one that answers. It is not a nicer version of the
// search: it is served by the endpoint over the whole record, which is the one
// thing the box beside it cannot do.
//
// The Hebrew opens on a Hebrew word deliberately. Measured in the browser: a
// line that opens on the name renders left to right inside a right to left
// page, because every one of these actors is an ASCII login and dir="auto"
// takes the direction from the first strong character.
export function byActorLine(name) {
  return [
    `${actorLabel(name, 'en')} is on the record, and the operator filter reads all of it rather than this page.`,
    `ברישום יש רשומות של ${actorLabel(name, 'he')}, וסינון המפעיל קורא את כל הרישום ולא רק את העמוד הזה.`,
  ];
}

export function byActorControl(name) {
  return [
    `Show every entry by ${actorLabel(name, 'en')}`,
    `הצגת כל הרשומות של ${actorLabel(name, 'he')}`,
  ];
}

function kindAbsentLine(kind, rest) {
  const label = KIND_LABELS[kind] || [kind, kind];
  return [
    `No entry of kind ${label[0]} is on the record. ${rest[0]} entries are, in other kinds.`,
    `אין ברישום אף רשומה מסוג ${label[1]}. יש בו ${rest[1]} רשומות מסוגים אחרים.`,
  ];
}

// The actor sentence has to know whether a kind is narrowing it too, because
// "nothing by them is on the record" is false when the truth is "nothing by them
// under this kind".
function actorAbsentLine(actor, kind, rest) {
  const who = [actorLabel(actor, 'en'), actorLabel(actor, 'he')];
  if (kind) {
    const label = KIND_LABELS[kind] || [kind, kind];
    return [
      `Nothing of kind ${label[0]} by ${who[0]} is on the record. ${rest[0]} entries are, in other kinds or by others.`,
      `אין ברישום אף רשומה מסוג ${label[1]} של ${who[1]}. יש בו ${rest[1]} רשומות מסוגים אחרים או של אחרים.`,
    ];
  }
  return [
    `Nothing by ${who[0]} is on the record. ${rest[0]} entries are, by others.`,
    `אין ברישום דבר של ${who[1]}. יש בו ${rest[1]} רשומות של אחרים.`,
  ];
}

function pastTheEndLine(matched) {
  return [
    `This page is past the last of the ${count(matched, 'en')} matching entries.`,
    `העמוד הזה נמצא אחרי האחרונה מבין ${count(matched, 'he')} הרשומות התואמות.`,
  ];
}

function searchedWholeLine(matched) {
  return [
    `Nothing matches that search. Every one of the ${count(matched, 'en')} entries this page holds was searched.`,
    `שום דבר לא תואם את החיפוש. כל ${count(matched, 'he')} הרשומות שבעמוד הזה נבדקו.`,
  ];
}

function filteredLine(rest) {
  return [
    `Nothing here matches those filters. ${rest[0]} entries are on the record.`,
    `שום דבר כאן לא תואם את הסינון. יש ${rest[1]} רשומות ברישום.`,
  ];
}

// Why the list is empty with no day window set, and which control reaches the
// rest. Six states, in the order that puts the most specific true statement
// first, and every figure in every one of them is the endpoint's own.
export function emptyPage(body, view) {
  const source = body || {};
  const set = view || {};
  const total = Number(source.total || 0);
  const matched = Number(source.matched || 0);
  const served = Number(source.served || 0);
  const counts = source.counts || {};
  const kind = String(set.kind || '');
  const actor = String(set.actor || '');
  const needle = String(set.needle || '').trim();
  const limit = Number(set.limit || 0);
  const wide = Number(set.wide || 0);
  const rest = [count(total, 'en'), count(total, 'he')];
  const still = { covers: null, actor: '', clear: false, wide: false, older: false, newest: false };
  if (!total) return { ...still, line: RECORD_EMPTY };
  if (kind && !Number(counts[kind] || 0)) return { ...still, line: kindAbsentLine(kind, rest), clear: true };
  if (actor && !matched) return { ...still, line: actorAbsentLine(actor, kind, rest), clear: true };
  // Nothing matched at all, so there is no page to be past the end of and no set
  // for the search to have covered part of. Neither figured sentence below can
  // be said over a set of nothing, and this one can.
  if (!matched) return { ...still, line: filteredLine(rest), clear: Boolean(kind || actor || needle) };
  if (!served) return { ...still, line: pastTheEndLine(matched), newest: true, clear: Boolean(kind || actor || needle) };
  if (!needle) return { ...still, line: filteredLine(rest), clear: Boolean(kind || actor) };
  // The page holds everything that matched, so the search really did read all of
  // it and "nothing matches" is a whole answer rather than a page's answer.
  if (!Number(source.older || 0) && !Number(source.newer || 0)) {
    return { ...still, line: searchedWholeLine(matched), clear: true };
  }
  return {
    line: SEARCH_MISS,
    covers: searchCoversLine(source, Boolean(kind || actor)),
    actor: actorForNeedle(source.actors, needle),
    clear: true,
    wide: Boolean(limit && wide && limit < wide && matched > limit),
    older: Boolean(source.next_before),
    newest: Number(source.newer || 0) > 0,
  };
}
