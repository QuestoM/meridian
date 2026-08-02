import React from 'react';
import { pageText } from '../shell/surface-helpers';

// A proposal's one-line summary, said in the reader's own language.
//
// The server's summary string is a record: it goes to the audit trail and back
// to the model, and it is written in English machine grammar. Printing it here
// put that record on the surface in both languages, under a Hebrew heading in
// one and under the approved English label in the other, where it still carried
// the retired word recompute. So the server sends the terms it built the
// sentence from (item.summary_terms = a code plus the values), and this reads
// them in whichever language the reader is in. Neither language is the fallback
// for the other: one branch per code carries both readings, so a code can never
// be sayable in Hebrew and unsayable in English.
//
// Nothing is translated by guessing: an unknown code or an unknown vocabulary
// token falls back to the exact token the server sent, isolated as ltr text,
// and an item with no terms at all falls back to the summary itself, which is
// what the calendar-event and agency proposals want because their summaries are
// already Hebrew. The rate-card note is the one string that stays as the server
// wrote it in both languages: it is measured disclosure, not a label.

const CONSTRAINT_EFFECTS = {
  fix_offset: { he: 'נעיצת ברייק בנקודת זמן קבועה', en: 'Pin a break at a fixed time' },
  offset_window: { he: 'נעיצת ברייק במרכז חלון זמן', en: 'Pin a break inside a time window' },
  pin_count: { he: 'קיבוע מספר הברייקים', en: 'Fix the number of breaks' },
  duration_range: { he: 'הגבלת אורך הברייק', en: 'Limit the break length' },
  gold: { he: 'סימון ברייקי זהב', en: 'Mark gold breaks' },
  forbid: { he: 'ללא ברייקים', en: 'No breaks' },
};

const CONSTRAINT_SCOPES = {
  programme: { he: 'בתוכנית', en: 'in the programme' },
  date: { he: 'בתאריך', en: 'on the date' },
  weekday: { he: 'ביום בשבוע', en: 'on the weekday' },
  channel: { he: 'בערוץ', en: 'on the channel' },
  always: { he: 'בכל רצועות השידור', en: 'across every broadcast strip' },
};

const OVERRIDE_KINDS = {
  pin: { he: 'קיבוע מספר הברייקים', en: 'Fix the number of breaks' },
  force: { he: 'מינימום ברייקים', en: 'Minimum breaks' },
  forbid: { he: 'ללא ברייקים', en: 'No breaks' },
  gold: { he: 'ברייקי זהב', en: 'Gold breaks' },
  lock: { he: 'נעילת ספוט', en: 'Lock a spot' },
  move: { he: 'העברת ספוט', en: 'Move a spot' },
};

const OVERRIDE_SCOPES = {
  segment: { he: 'ברצועת שידור', en: 'on the broadcast strip' },
  spot: { he: 'בספוט', en: 'on the spot' },
};

function Token({ value }) {
  return <bdi dir="ltr">{String(value)}</bdi>;
}

// A list of identifiers, each isolated so a Hebrew sentence around them stays
// readable and no key is reordered by the bidi algorithm.
function Tokens({ values }) {
  return (
    <>
      {values.map((value, index) => (
        <React.Fragment key={`${value}-${index}`}>
          {index ? ', ' : ''}
          <Token value={value} />
        </React.Fragment>
      ))}
    </>
  );
}

function word(table, token, locale) {
  const key = String(token || '');
  const pair = table[key];
  return pair ? <>{pageText(locale, pair.en, pair.he)}</> : <Token value={key || '?'} />;
}

// The reading of one terms object in the reader's language, or null when the
// code is unknown. Both languages come out of the same branch on purpose.
function say(terms, locale) {
  const t = (en, he) => pageText(locale, en, he);
  const fields = Array.isArray(terms.fields) ? terms.fields.map(String) : [];
  if (terms.code === 'settings') {
    return <>{t('Settings: ', 'הגדרות: ')}<Tokens values={fields} /></>;
  }
  if (terms.code === 'recompute') {
    if (terms.scope === 'full') return <>{t('Run the plan for the whole week', 'הרצת התוכנית על כל השבוע')}</>;
    const days = Array.isArray(terms.days) ? terms.days.map(String) : [];
    return <>{t('Run the plan for these days: ', 'הרצת התוכנית לימים: ')}<Tokens values={days} /></>;
  }
  if (terms.code === 'constraint') {
    return (
      <>
        {t('Restriction: ', 'הגבלה: ')}
        {word(CONSTRAINT_EFFECTS, terms.effect, locale)}
        {' '}
        {word(CONSTRAINT_SCOPES, terms.scope_type, locale)}
        {terms.scope_value ? <> <Token value={terms.scope_value} /></> : null}
        {terms.predicate ? <>{t(', with a further condition', ', עם תנאי נוסף')}</> : null}
      </>
    );
  }
  if (terms.code === 'override') {
    return (
      <>
        {word(OVERRIDE_KINDS, terms.kind, locale)}
        {' '}
        {word(OVERRIDE_SCOPES, terms.scope, locale)}
        {terms.target_id ? <> <Token value={terms.target_id} /></> : null}
        {terms.value ? <>{t(', value ', ', ערך ')}<Token value={terms.value} /></> : null}
      </>
    );
  }
  if (terms.code === 'pricing') {
    const keys = Array.isArray(terms.keys) ? terms.keys.map(String) : [];
    return (
      <>
        {t('Pricing: edit ', 'תמחור: עריכת ')}
        <Tokens values={keys} />
        {terms.note ? <>{'. '}<span dir="auto">{String(terms.note)}</span></> : null}
      </>
    );
  }
  if (terms.code === 'advertiser') {
    return (
      <>
        {terms.action === 'create'
          ? t('Advertiser: create ', 'מפרסם: יצירת ')
          : t('Advertiser: update ', 'מפרסם: עדכון ')}
        <Token value={terms.name} />
        {fields.length ? <>{' ('}<Tokens values={fields} />{')'}</> : null}
      </>
    );
  }
  return null;
}

// True when this item can be said from its terms in the given language.
// Exported so a test can assert every code the server emits has a reading here,
// in both languages rather than in Hebrew alone.
export function hasReading(item, locale) {
  const terms = item && item.summary_terms;
  if (!terms || typeof terms !== 'object' || !terms.code) return false;
  return say(terms, locale) !== null;
}

export default function ProposalSummary({ item, locale, className }) {
  const summary = item && item.summary ? String(item.summary) : '';
  const terms = item && item.summary_terms && typeof item.summary_terms === 'object'
    ? item.summary_terms
    : null;
  const said = terms ? say(terms, locale) : null;
  if (!said && !summary) return null;
  return <p className={className} dir="auto">{said || summary}</p>;
}
