// The agreement as an OBJECT: what level it sits at, where it is in its
// lifecycle, who signed it, how long it runs, and how its document was read.
//
// trade-terms.js is the twin of the term registry and term-language.js turns one
// term's parameters into a sentence. This file is the layer around both, holding
// the vocabulary that belongs to the agreement itself rather than to any clause
// inside it. The agreement as an EFFECT — mechanisms, blockers, reviewer verdicts,
// alarms, money lines — is its sibling trade-effects.js, re-exported at the bottom
// so a caller resolves one vocabulary either way.
//
// TWO RULES DECIDE WHAT GOES IN HERE.
//
// The backend's sentence wins. Every Hebrew sentence the engine authors is printed
// as it arrives; what this file adds is the frame around it — the label on a
// badge, the name of a lifecycle state, the reading of a window. Nothing here
// re-derives a fact the engine already decided.
//
// A tone is never the whole message. Every entry carries copy, and the surface
// pairs it with an icon, so a state is legible without colour.

import { pageText } from '../shell/format';

// ------------------------------------------------------------------ lifecycle

// The three the store accepts (kairos_api/trade_store.py LEVELS). An amendment
// or an appendix is not a fourth level: it is an agreement at one of these three
// levels carrying a parent, which is how precedence between them is decided.
const LEVELS = {
  agency_framework: { he: 'הסכם מסגרת עם סוכנות', en: 'Agency framework' },
  advertiser: { he: 'הסכם מפרסם', en: 'Advertiser agreement' },
  campaign: { he: 'הסכם קמפיין', en: 'Campaign agreement' },
};

export function levelLabel(level, locale) {
  const entry = LEVELS[String(level || '')];
  if (!entry) return String(level || '');
  return locale === 'he' ? entry.he : entry.en;
}

export function levelOptions(locale) {
  return Object.entries(LEVELS).map(([value, entry]) => ({
    value,
    label: locale === 'he' ? entry.he : entry.en,
  }));
}

// The lifecycle, with the tone that says whether the agreement is acting on the
// business. Only an approved agreement can bind a rule, and the vocabulary says
// so rather than leaving four shades of grey to be guessed at.
const STATUSES = {
  draft: { he: 'טיוטה', en: 'Draft', tone: 'neutral' },
  in_review: { he: 'בסקירה', en: 'In review', tone: 'info' },
  approved: { he: 'מאושר', en: 'Approved', tone: 'positive' },
  superseded: { he: 'הוחלף', en: 'Superseded', tone: 'neutral' },
  expired: { he: 'פג תוקף', en: 'Expired', tone: 'warning' },
  withdrawn: { he: 'בוטל', en: 'Withdrawn', tone: 'danger' },
};

export function statusLabel(status, locale) {
  const entry = STATUSES[String(status || '')];
  if (!entry) return String(status || '');
  return locale === 'he' ? entry.he : entry.en;
}

export function statusTone(status) {
  const entry = STATUSES[String(status || '')];
  return entry ? entry.tone : 'neutral';
}

const COUNTERPARTY_KINDS = {
  agency: { he: 'סוכנות', en: 'Agency' },
  advertiser: { he: 'מפרסם', en: 'Advertiser' },
  advertiser_via_agency: { he: 'מפרסם באמצעות סוכנות', en: 'Advertiser via agency' },
};

export function counterpartyKind(kind, locale) {
  const entry = COUNTERPARTY_KINDS[String(kind || '')];
  // A kind this vocabulary does not know is a vocabulary gap, not a display
  // string: the raw enum key reached a card once (advertiser_via_agency) and
  // read as debugging output beside every properly named party.
  if (!entry) return locale === 'he' ? 'צד להסכם' : 'Counterparty';
  return locale === 'he' ? entry.he : entry.en;
}

// The counterparty block arrives in two shapes across the corpus and the API:
// {kind, name} from the store's own create, and {counterparty_type, agency} from
// an extraction. Both are read here so no surface has to know which it got.
// The advertiser outranks the agency when both are present: the advertiser IS
// the party, the agency represents it — and the agency field may hold a bare
// storage id (AGY_04) that must never stand in as the counterparty's name.
export function counterpartyName(counterparty) {
  if (!counterparty || typeof counterparty !== 'object') return '';
  return String(counterparty.name || counterparty.advertiser || counterparty.agency || '');
}

export function counterpartyKindOf(counterparty) {
  if (!counterparty || typeof counterparty !== 'object') return '';
  return String(counterparty.kind || counterparty.counterparty_type || '');
}

// ------------------------------------------------------------ how it was read

// HOW THE DOCUMENT'S TEXT WAS OBTAINED, which is provenance and not trivia: a
// digital text layer is read exactly, and a scanned page is read by a vision
// model with a different and larger error profile. A reviewer checking a quote
// against a page needs to know which of the two they are auditing.
//
// The engine's values are `digital` and `scanned-vision` (kairos/trade/ingest.py).
// Measured before this table existed: the surface compared against 'scanned',
// which never matches, so every scanned contract in the corpus was labelled as
// read from digital text — the confident claim being the wrong one. An
// unrecognised route now says it is unrecognised rather than defaulting to the
// reassuring answer.
const INGEST_ROUTES = {
  digital: {
    he: 'נקרא משכבת טקסט דיגיטלית',
    en: 'Read from a digital text layer',
  },
  'scanned-vision': {
    he: 'עמוד סרוק, נקרא במודל ראייה',
    en: 'A scanned page, read by a vision model',
  },
};

export function ingestRouteLabel(route, locale) {
  const key = String(route || '');
  const entry = INGEST_ROUTES[key];
  if (entry) return locale === 'he' ? entry.he : entry.en;
  if (!key) {
    return pageText(locale, 'The reading route was not recorded', 'נתיב הקריאה לא נרשם');
  }
  return pageText(locale, `Reading route: ${key}`, `נתיב קריאה: ${key}`);
}

export function ingestRouteTone(route) {
  return String(route || '') === 'scanned-vision' ? 'warning' : 'neutral';
}

// --------------------------------------------------------------- the window

// EVERY AGREEMENT HAS AN END DATE, and the store enforces it: an obligation with
// no closing date has no measurement window, so its pace, its projection and its
// alarm are all undefined. An agreement the parties meant to run until somebody
// cancels is therefore stored against a sentinel far-future date with an
// `open_ended` flag, and this reader turns that back into the sentence the
// parties actually agreed rather than printing a literal 2099 deadline.
//
// The window arrives under two key pairs — `starts_on`/`ends_on` from the store
// and `from`/`to` from an extraction — so both are read here and nowhere else.
export const OPEN_ENDED_UNTIL = '2099-12-31';

export function windowOf(window) {
  if (!window || typeof window !== 'object') return { from: '', to: '', openEnded: false };
  const from = String(window.starts_on || window.from || '');
  const to = String(window.ends_on || window.to || '');
  const openEnded = Boolean(window.open_ended) || to === OPEN_ENDED_UNTIL;
  return { from, to: openEnded ? '' : to, openEnded };
}

export function openEndedLabel(locale) {
  return pageText(
    locale,
    'Open-ended: it runs until one side cancels',
    'ללא מועד סיום: בתוקף עד שאחד הצדדים יבטל',
  );
}

// The review and measurement half of this vocabulary — mechanisms, blockers,
// reviewer verdicts, alarms and the money lines of a simulation — lives in
// trade-effects.js and is re-exported here, so every caller keeps resolving one
// vocabulary regardless of which half a word belongs to.
export * from './trade-effects';
