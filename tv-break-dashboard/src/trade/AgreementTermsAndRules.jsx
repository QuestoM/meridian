import React from 'react';
import { Card, CardBody, EmptyState, Status } from '../studio';
import { Lock, ScrollText } from 'lucide-react';
import { Code, Figure, Name, Prose } from '../shell/bidi';
import { formatNumber, pageText } from '../shell/format';
import { FAMILY_ORDER, familyName, termName } from './trade-terms';
import { mechanismLabel, mechanismTone } from './trade-vocabulary';

// The terms the agreement holds, grouped the way a contract is organised, and the
// live rules it currently owns in the engine's own stores.
//
// TWO DIFFERENT QUESTIONS, deliberately side by side. "What does this agreement
// say" is answered by the term list. "What is the machinery actually doing about
// it right now" is answered by the bound-rule list, read from the live pricing,
// frequency and settlement stores. The gap between the two lists is the honest
// measure of how much of the agreement is really wired, and putting them on one
// screen is what makes that gap impossible to miss.
//
// THE BASIS IS STATED. The term list comes from the reviewed state of the
// attached document, which for an approved agreement is frozen — the documents
// stop being editable at approval, so the reviewed state and the approved termset
// are the same set. That is said on the surface rather than assumed.

function TermRow({ term, locale }) {
  return (
    <li className="trd-term-row" data-mechanism={term.mechanism}>
      <span className="trd-term-row-name">
        <Name>{term.term_name_he && locale === 'he' ? term.term_name_he : termName(term.term_id, locale)}</Name>
        <Code className="trd-id-chip">{term.instance_id}</Code>
      </span>
      <Status status={mechanismTone(term.mechanism)}>
        {mechanismLabel(term.mechanism, locale, term.mechanism_he)}
      </Status>
      {term.sentence_he ? <Prose as="span" className="trd-term-row-sentence">{term.sentence_he}</Prose> : null}
      {term.will_not_act_reasons && term.will_not_act_reasons.length > 0 ? (
        <Prose as="span" className="trd-term-row-inert">{term.will_not_act_reasons[0]}</Prose>
      ) : null}
    </li>
  );
}

export function AgreementTermsByFamily({ effects, locale }) {
  const terms = (effects && effects.terms) || [];
  if (terms.length === 0) {
    // `unread` arrives when the proposal route said this document has no
    // extraction yet — a known stage, so the sentence states it instead of
    // hedging between two possibilities.
    const unread = Boolean(effects && effects.unread);
    return (
      <EmptyState
        title={pageText(locale, 'No terms are held for this agreement', 'לא נשמרים מונחים להסכם הזה')}
        description={unread
          ? pageText(
            locale,
            'The document is attached but has not been read yet. Opening the review runs the reading.',
            'המסמך מצורף אך טרם נקרא. פתיחת הסקירה מריצה את הקריאה.',
          )
          : pageText(
            locale,
            'Either the document has not been read yet, or every proposed term was rejected in review.',
            'או שהמסמך עדיין לא נקרא, או שכל מונח מוצע נדחה בסקירה.',
          )}
      />
    );
  }
  const byFamily = new Map();
  terms.forEach((term) => {
    const key = term.family || 'A';
    if (!byFamily.has(key)) byFamily.set(key, []);
    byFamily.get(key).push(term);
  });
  const groups = FAMILY_ORDER
    .filter((family) => byFamily.has(family))
    .map((family) => ({ family, terms: byFamily.get(family) }));

  return (
    <section className="trd-families" aria-label={pageText(locale, 'The terms this agreement holds', 'המונחים שההסכם מחזיק')}>
      <div className="trd-pane-head">
        <h4>
          <ScrollText size={16} aria-hidden="true" />
          {pageText(locale, 'The terms this agreement holds', 'המונחים שההסכם מחזיק')}
        </h4>
        <span className="trd-chip-quiet">
          {pageText(
            locale,
            `${formatNumber(terms.length, locale)} terms`,
            `${formatNumber(terms.length, locale)} מונחים`,
          )}
        </span>
      </div>
      <p className="trd-field-hint">
        {pageText(
          locale,
          'Read from the reviewed state of the attached document, which an approved agreement can no longer change. Each line says what the engine will do about that term, not what the clause hoped for.',
          'נקרא מהמצב הסקור של המסמך המצורף, שהסכם מאושר אינו יכול לשנות עוד. כל שורה אומרת מה המנוע יעשה עם המונח, לא מה הסעיף קיווה.',
        )}
      </p>
      {groups.map((group) => (
        <Card key={group.family} className="trd-family">
          <CardBody>
            <h5>
              {familyName(group.family, locale)}
              <span className="trd-group-count">{formatNumber(group.terms.length, locale)}</span>
            </h5>
            <ul className="trd-term-rows">
              {group.terms.map((term) => (
                <TermRow key={term.instance_id} term={term} locale={locale} />
              ))}
            </ul>
          </CardBody>
        </Card>
      ))}
    </section>
  );
}

const STORE_NAMES = {
  advertiser_conditions: { he: 'תנאי מפרסם (תמחור ושיבוץ)', en: 'Advertiser conditions (pricing and placement)' },
  agency_conditions: { he: 'תנאי סוכנות (תמחור ושיבוץ)', en: 'Agency conditions (pricing and placement)' },
  frequency_rules: { he: 'כללי תדירות והפרדה', en: 'Frequency and separation rules' },
};

function storeName(key, locale) {
  const entry = STORE_NAMES[key];
  if (!entry) return String(key || '');
  return locale === 'he' ? entry.he : entry.en;
}

export function BoundRules({ boundRules, locale }) {
  const entries = Object.entries(boundRules || {});
  const total = entries.reduce((sum, [, rows]) => sum + (Array.isArray(rows) ? rows.length : 0), 0);
  return (
    <section className="trd-bound" aria-label={pageText(locale, 'Live rules this agreement owns', 'כללים פעילים שההסכם מחזיק')}>
      <div className="trd-pane-head">
        <h4>
          <Lock size={16} aria-hidden="true" />
          {pageText(locale, 'Live rules this agreement owns', 'כללים פעילים שההסכם מחזיק')}
        </h4>
        <span className="trd-chip-quiet">
          {pageText(
            locale,
            `${formatNumber(total, locale)} rows`,
            `${formatNumber(total, locale)} שורות`,
          )}
        </span>
      </div>
      {total === 0 ? (
        <EmptyState
          title={pageText(locale, 'This agreement writes no live rule', 'ההסכם הזה אינו כותב כלל פעיל')}
          description={pageText(
            locale,
            'Nothing in the pricing, frequency or settlement stores currently belongs to it. Either it is not approved yet, or every term it holds is one the engine records rather than acts on. The term list above says which.',
            'דבר במאגרי התמחור, התדירות וההתחשבנות אינו שייך לו כרגע. או שהוא עדיין לא אושר, או שכל מונח שהוא מחזיק הוא מונח שהמנוע רושם ולא פועל לפיו. רשימת המונחים שלמעלה אומרת מה מהשניים.',
          )}
        />
      ) : (
        entries.map(([store, rows]) => (
          <Card key={store} className="trd-bound-store">
            <CardBody>
              <h5>
                {storeName(store, locale)}
                <span className="trd-group-count">{formatNumber(rows.length, locale)}</span>
              </h5>
              <ul className="trd-bound-list">
                {rows.map((row) => (
                  <li key={row.rule_id}>
                    <Code className="trd-id-chip">{row.rule_id}</Code>
                    {row.effect ? <span className="trd-chip-quiet">{row.effect}</span> : null}
                    {row.limit_type ? <span className="trd-chip-quiet">{row.limit_type}</span> : null}
                    {row.value !== undefined && row.value !== '' ? (
                      <Figure>{String(row.value)}</Figure>
                    ) : null}
                    {row.notes ? <Prose as="span" className="trd-meta-note">{row.notes}</Prose> : null}
                  </li>
                ))}
              </ul>
            </CardBody>
          </Card>
        ))
      )}
    </section>
  );
}
