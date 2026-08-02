// How far back this page reaches, what it holds, what it leaves out, and how to
// reach further.
//
// A page is a window on the record, never the record. The endpoint serves at
// most 500 entries and always the newest of whatever matched, and until this
// round the window was nailed to the newest end with nothing on the surface
// saying so. Measured on the running instance: the change filter matched 2,027
// entries, the page served 500 of them spanning one afternoon, and the footer
// counted rows without ever mentioning the 1,527 it had dropped. A compliance
// owner reading "2,011 changes recorded since 15 July" above a list that begins
// this afternoon could only conclude that nothing changed before today.
//
// The rule is the one this destination already applies to the run count: a
// number the product may not print must say so rather than print a reassuring
// one. Here the product can print it, so it prints all of it, and it carries a
// control that reaches the part the page does not hold.
//
// This is plain JavaScript so the sentences can be executed by a test rather
// than grepped, which is the pattern history-runs.js established here.

// The extension is explicit because this module is executed by node in a test,
// which is the point of it being plain JavaScript, and node resolves no other way.
import { KIND_LABELS, actorLabel } from './history-labels.js';

// What the payload says about where this page sits in the matched set. Every
// figure is the endpoint's own; nothing here counts anything.
export function reachState(body) {
  const source = body || {};
  const served = Number(source.served || 0);
  const newer = Number(source.newer || 0);
  const older = Number(source.older || 0);
  const matched = Number(source.matched || 0);
  return {
    matched,
    served,
    newer,
    older,
    from: served ? newer + 1 : 0,
    to: newer + served,
    cursor: String(source.next_before || ''),
    // A window is in force when the page is not the whole matched set. That is
    // the only state in which any of this has to be said at all.
    windowed: newer > 0 || older > 0,
    paged: newer > 0,
  };
}

function count(value, locale) {
  return Number(value || 0).toLocaleString(locale === 'he' ? 'he-IL' : 'en-GB');
}

// Where the window sits, in the matched set, in one sentence.
export function reachLine(reach) {
  const en = `Entries ${count(reach.from, 'en')} to ${count(reach.to, 'en')} of ${count(reach.matched, 'en')} matching.`;
  const he = `רשומות ${count(reach.from, 'he')} עד ${count(reach.to, 'he')} מתוך ${count(reach.matched, 'he')} תואמות.`;
  return [en, he];
}

// What the page does not hold. Printed whenever it holds less than everything.
export function olderLine(reach) {
  const en = `${count(reach.older, 'en')} matching entries are older than this page.`;
  const he = `${count(reach.older, 'he')} רשומות תואמות ישנות יותר מהעמוד הזה.`;
  return [en, he];
}

// The search runs over the loaded page, so with a page that holds part of the
// set, a search that finds nothing is not evidence that nothing is there.
export const SEARCH_SCOPE = [
  'The search runs over this page, not over the entries older than it.',
  'החיפוש פועל על העמוד הזה, לא על הרשומות הישנות ממנו.',
];

export const OLDER_CONTROL = ['Older', 'ישנות יותר'];
export const NEWEST_CONTROL = ['Back to the newest', 'חזרה לחדשות ביותר'];

// The day window. Two controls rather than one, because the compliance question
// is a day and not a direction: from and up to, both inclusive, both read in the
// broadcast zone the list is already grouped by.
export const FROM_CONTROL = ['From', 'מיום'];
export const UNTIL_CONTROL = ['Up to', 'עד ליום'];
export const FROM_HINT = ['Show entries recorded on this day or after it.', 'הצגת רשומות שנרשמו ביום הזה או אחריו.'];
export const UNTIL_HINT = ['Show entries recorded on this day or before it.', 'הצגת רשומות שנרשמו ביום הזה או לפניו.'];
export const DAYS_CLEAR = ['Clear the days', 'ניקוי הימים'];
export const DAY_JUMP = ['Go to that day', 'מעבר ליום ההוא'];

// Nothing was recorded in the days the reader set. The one sentence in this
// module that says the record itself is empty, and it may only be said when the
// window holds nothing at all and the record reaches those days in the first
// place.
export const EMPTY_WINDOW = [
  'Nothing was recorded in those days.',
  'לא נרשם דבר בימים האלה.',
];

// How far back the record reaches, and what drops the rest.
//
// Two of the four records this page merges are bounded. Measured on disk on
// 2026-08-01 at 19:45 UTC: the request recorder held 5,227 lines, the oldest
// stamped 14:42 that day, and exactly 200 restore points survived, the oldest
// stamped 11:01. Asked for 20 July, every tab read 0 and the page said nothing
// was recorded in those days. Nothing that survives was. The page could not
// tell the difference and said the reassuring one.
//
// So the record's own start rides every read, and it is printed under every
// list rather than only under an empty one. The doctrine is the one this
// destination already applies to the guardrail store in HistorySince.jsx: the
// day the record starts is printed beside the count, so nobody mistakes
// "nothing recorded" for "nothing happened before the record existed".

// The name each bounded store carries in a sentence, and what it keeps. A store
// that keeps everything has no clause here, because then its start is simply the
// first thing ever recorded in it and there is nothing to explain.
const KEEPS = {
  changes: (en, he) => [
    `the request recorder keeps the newest ${en} lines`,
    `רישום הבקשות שומר את ${he} השורות האחרונות`,
  ],
  restore_points: (en, he) => [
    `the version store keeps the newest ${en} restore points`,
    `מאגר הגרסאות שומר את ${he} נקודות השחזור האחרונות`,
  ],
};

// Which record each kind is kept in, so a kind whose own record was pruned past
// the window says so instead of reporting that nothing of it happened. Restores
// are deliberately absent: that record is append-only and complete, so a restore
// missing from a day is genuinely missing.
const KIND_SOURCE = {
  change: 'changes',
  preview: 'changes',
  sign_in: 'changes',
  restore_point: 'restore_points',
  run: 'runs',
};

function sourceOf(body, name) {
  return ((body || {}).sources || {})[name] || {};
}

// The clause for every bounded store in the payload, in both languages. Nothing
// is counted here: keeps is the store's own constant, travelling with the read.
export function retentionClauses(body) {
  return Object.keys(KEEPS)
    .map((name) => {
      const keeps = Number((sourceOf(body, name).retention || {}).keeps || 0);
      return keeps ? KEEPS[name](count(keeps, 'en'), count(keeps, 'he')) : null;
    })
    .filter(Boolean);
}

function joined(clauses, index) {
  const parts = clauses.map((clause) => clause[index]);
  if (parts.length < 2) return parts[0] || '';
  const last = parts[parts.length - 1];
  const rest = parts.slice(0, -1).join(', ');
  return index === 0 ? `${rest} and ${last}` : `${rest}, ו${last}`;
}

// A day window is out of a record's reach when every day in it is older than the
// day that record starts on. Only a bounded store can be out of reach: one that
// keeps everything and starts later simply has nothing before it, which is a
// different sentence and a true one.
function beyond(body, starts, bounded) {
  const until = windowDays(body).until;
  return Boolean(bounded && starts && until && until < starts);
}

export function windowOutOfReach(body) {
  return beyond(body, String((body || {}).record_starts || ''), retentionClauses(body).length > 0);
}

export function kindOutOfReach(body, kind) {
  const source = sourceOf(body, KIND_SOURCE[kind]);
  const retention = source.retention || {};
  return beyond(body, String(source.starts || ''), Boolean(retention.pruned && retention.keeps));
}

// Where the record starts, printed under every list on this destination whether
// it is full or empty, because the answer does not change with the list.
export function recordStartLine(body) {
  const starts = String((body || {}).record_starts || '');
  if (!starts) return null;
  const clauses = retentionClauses(body);
  if (!clauses.length) return [`The record starts on ${starts}.`, `הרישום מתחיל ב-${starts}.`];
  return [
    `The record starts on ${starts}; ${joined(clauses, 0)}.`,
    `הרישום מתחיל ב-${starts}; ${joined(clauses, 1)}.`,
  ];
}

// The window is entirely before the record starts, so this page holds no
// evidence about those days either way. It is the sentence EMPTY_WINDOW was
// firing in place of.
export function outOfReachLine(body) {
  const starts = String((body || {}).record_starts || '');
  const clauses = retentionClauses(body);
  const en = clauses.length ? `It starts on ${starts}; ${joined(clauses, 0)}.` : `It starts on ${starts}.`;
  const he = clauses.length ? `הוא מתחיל ב-${starts}; ${joined(clauses, 1)}.` : `הוא מתחיל ב-${starts}.`;
  return [`The record does not reach those days. ${en}`, `הרישום אינו מגיע לימים האלה. ${he}`];
}

// The same thing for one kind, when the window holds entries of other kinds but
// this kind's own record was pruned past it.
export function kindOutOfReachLine(body, kind) {
  const label = KIND_LABELS[kind] || [kind, kind];
  const source = sourceOf(body, KIND_SOURCE[kind]);
  const starts = String(source.starts || '');
  const keeps = Number((source.retention || {}).keeps || 0);
  const clause = keeps && KEEPS[KIND_SOURCE[kind]] ? KEEPS[KIND_SOURCE[kind]](count(keeps, 'en'), count(keeps, 'he')) : null;
  const en = clause ? `It starts on ${starts}; ${clause[0]}.` : `It starts on ${starts}.`;
  const he = clause ? `הוא מתחיל ב-${starts}; ${clause[1]}.` : `הוא מתחיל ב-${starts}.`;
  return [
    `The record of kind ${label[0]} does not reach those days. ${en}`,
    `הרישום מסוג ${label[1]} אינו מגיע לימים האלה. ${he}`,
  ];
}

// The day the record behind the attestation count starts on, printed beside that
// count. Measured before this line existed: the strip read "2,562 changes and
// points recorded" for a window opening on 14 June, beside a sentence naming
// 2026-06-14 as the start of a different record altogether, over five hours of
// surviving evidence.
export function attestationStartLine(body) {
  const starts = String((body || {}).record_starts || '');
  if (!starts) return null;
  const day = String((body || {}).day || '');
  if (day && day < starts) {
    return [
      `The record behind this count starts on ${starts}, so the days before it hold no evidence either way.`,
      `הרישום שמאחורי הספירה הזאת מתחיל ב-${starts}, ולכן הימים שלפניו אינם ראיה לכאן או לכאן.`,
    ];
  }
  return [
    `The record behind this count starts on ${starts}.`,
    `הרישום שמאחורי הספירה הזאת מתחיל ב-${starts}.`,
  ];
}

// Why the list under a day window is empty when the record in those days is not.
//
// Measured on the running instance before this fix, two clicks from the landing
// state: with "Up to" set to 28/07/2026 and the Change tab selected, the tab row
// read Everything 30, Change 0, Run 29, Restore 1, and the list under it said
// nothing was recorded in those days. Thirty entries were, and the page had
// every figure in hand. A compliance owner who narrows to a day and to changes
// read that the record was empty for those days.
//
// So the sentence is chosen from the payload's own counts rather than from the
// presence of a window. The record is empty in those days only when the window
// itself holds nothing. Otherwise the page names the control that emptied the
// list and prints what dropping it would reveal, which is the figure the
// Everything tab is already showing beside it.
//
// And the record is only empty in those days if it reaches them at all. Measured
// live before that guard: "Up to" 20/07/2026 read 0 on every tab and the page
// said nothing was recorded in those days, while the oldest surviving entry in
// the whole record was stamped 26 July. Both bounded stores had pruned past the
// question, and the page had the figures to know it.
//
// Nothing here counts anything: window_total, counts, matched, served and
// record_starts are all the endpoint's own. While the run log is withheld or
// unreadable they hold no run, which the run tab and the footer say in their own
// words.
export function emptyWindow(body, filters) {
  const source = body || {};
  const set = filters || {};
  const held = source.window_total === undefined ? Number(source.total || 0) : Number(source.window_total);
  const matched = Number(source.matched || 0);
  const served = Number(source.served || 0);
  const counts = source.counts || {};
  const kind = String(set.kind || '');
  const actor = String(set.actor || '');
  const needle = String(set.needle || '');
  if (!held) {
    if (windowOutOfReach(source)) {
      return { line: outOfReachLine(source), clear: false, scope: false, reach: true };
    }
    return { line: EMPTY_WINDOW, clear: false, scope: false, reach: false };
  }
  const rest = [count(held, 'en'), count(held, 'he')];
  if (kind && !Number(counts[kind] || 0)) {
    if (kindOutOfReach(source, kind)) {
      return { line: kindOutOfReachLine(source, kind), clear: true, scope: false, reach: true };
    }
    const label = KIND_LABELS[kind] || [kind, kind];
    return {
      line: [
        `No entry of kind ${label[0]} was recorded in those days. ${rest[0]} entries were, in other kinds.`,
        `בימים האלה לא נרשמה אף רשומה מסוג ${label[1]}. נרשמו ${rest[1]} רשומות מסוגים אחרים.`,
      ],
      clear: true, scope: false, reach: false,
    };
  }
  if (actor && !matched) {
    return {
      line: [
        `Nothing by ${actorLabel(actor, 'en')} was recorded in those days. ${rest[0]} entries were, by others.`,
        `בימים האלה לא נרשם דבר על ידי ${actorLabel(actor, 'he')}. נרשמו ${rest[1]} רשומות על ידי אחרים.`,
      ],
      clear: true, scope: false, reach: false,
    };
  }
  if (!served) {
    return {
      line: [
        `This page is past the last of the ${count(matched, 'en')} entries matching in those days.`,
        `העמוד הזה נמצא אחרי האחרונה מבין ${count(matched, 'he')} הרשומות התואמות בימים האלה.`,
      ],
      clear: false, scope: false, reach: false,
    };
  }
  if (needle) {
    return {
      line: [
        `Nothing on this page matches that search. It holds ${count(served, 'en')} of the ${count(matched, 'en')} entries matching in those days.`,
        `שום דבר בעמוד הזה לא תואם את החיפוש. יש בו ${count(served, 'he')} מתוך ${count(matched, 'he')} הרשומות התואמות בימים האלה.`,
      ],
      clear: true, scope: true, reach: false,
    };
  }
  return {
    line: [
      `Nothing here matches those filters. ${rest[0]} entries were recorded in those days.`,
      `שום דבר כאן לא תואם את הסינון. נרשמו ${rest[1]} רשומות בימים האלה.`,
    ],
    clear: true, scope: false, reach: false,
  };
}

// What the request recorder contributed: how large that store is, how much of it
// this reader may see, and why neither figure is the Change tab beside them.
//
// Measured on the running instance: the footer read "Changes: 5,088" from the
// recorder's own line count while the Change tab read 2,034, under the same
// word, while the neighbouring restore points and runs both matched their tabs.
// One recorded line becomes a change, a preview or a sign-in, so the line names
// the source rather than one of the three kinds it produces.
//
// Then the same sentence printed the reader's own slice under the store's name.
// Measured live on 2026-08-01 over a recorder holding 5,261 lines: signed in as
// an operator it read "The request recorder holds 1 line" directly beside "the
// request recorder keeps the newest 5,000 lines"; as a viewer, 4; as the admin,
// 5,261. A store that keeps 5,000 and holds 1 is a self-contradiction printed in
// one paragraph. The payload now carries both figures and the rule that produced
// the smaller one, so the sentence says which is which and neither can be read
// as the other.
export function changesSourceLine(changes) {
  const source = changes || {};
  const records = Number(source.records || 0);
  const held = [count(records, 'en'), count(records, 'he')];
  const mine = Number(source.in_scope || 0);
  if (source.in_scope === undefined || mine === records) {
    return [
      `The request recorder holds ${held[0]} lines, which become changes, previews and sign-ins.`,
      `רישום הבקשות מחזיק ${held[1]} שורות, שמהן נגזרים שינויים, תצוגות מקדימות וכניסות.`,
    ];
  }
  const slice = [count(mine, 'en'), count(mine, 'he')];
  if (String(source.scope || '') === 'self') {
    return [
      `The request recorder holds ${held[0]} lines, of which ${slice[0]} ${mine === 1 ? 'is' : 'are'} yours; each line is a change, a preview or a sign-in.`,
      `רישום הבקשות מחזיק ${held[1]} שורות, ומתוכן ${slice[1]} שלכם; כל שורה היא שינוי, תצוגה מקדימה או כניסה.`,
    ];
  }
  // Every line is this reader's to see and some of them did not parse, which is
  // the only other way the two figures part company. It is rare and it is still
  // a difference between the store and the page, so it is said rather than hidden.
  return [
    `The request recorder holds ${held[0]} lines, of which this read could use ${slice[0]}; each line is a change, a preview or a sign-in.`,
    `רישום הבקשות מחזיק ${held[1]} שורות, ומתוכן הקריאה הזאת יכלה להשתמש ב-${slice[1]}; כל שורה היא שינוי, תצוגה מקדימה או כניסה.`,
  ];
}

// Which days the window is on, as the endpoint echoes them back.
export function windowDays(body) {
  const window = (body || {}).window || {};
  return { from: String(window.since || ''), until: String(window.until || '') };
}

export const CLEAR_FILTERS = ['Clear the filters', 'ניקוי הסינון'];

// Why the entry a link asked for is not on this page. One of the four reasons
// history-address.js decides between, in the reader's own language.
export function missedLine(missed, points, limit) {
  if (missed === 'point_gone') {
    return [`That restore point is not among the ${points} points on record, so it cannot be opened here.`, `נקודת השחזור הזו אינה בין ${points} הנקודות שברישום, ולכן לא ניתן לפתוח אותה כאן.`];
  }
  if (missed === 'filtered') {
    return ['The entry this link points to is not in the filtered list.', 'הרשומה שהקישור מצביע עליה אינה ברשימה המסוננת.'];
  }
  if (missed === 'paged_out') {
    return [`The entry this link points to is older than the ${limit} entries on this page.`, `הרשומה שהקישור מצביע עליה ישנה יותר מ-${limit} הרשומות שבעמוד הזה.`];
  }
  return ['The entry this link points to is not in the record.', 'הרשומה שהקישור מצביע עליה אינה ברישום.'];
}
