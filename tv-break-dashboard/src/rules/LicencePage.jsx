import React, { useCallback, useEffect, useState } from 'react';
import { Button } from '@mui/material';
import { CalendarClock, ExternalLink, ShieldCheck, Tv } from 'lucide-react';
import { pageText } from '../shell/format';
import DateField from '../shell/DateField';
import { payloadCanEdit, WALLS } from '../session.js';
import LicenceLimits from './LicenceLimits';
import {
  fetchAttestation,
  limitLabel,
  pairLabel,
  recordGuardrailChange,
  refusalSentence,
  unitLabel,
  valuePair,
} from './rules-lib';

// The compliance owner's whole job on one screen: the seven checks with what was
// observed against what is allowed, the licence those limits came from with its
// date and its source, and the evidence that nothing moved since they last
// signed. An empty change list is the evidence, so it is stated in words rather
// than left as a blank for a reader to interpret.

function CheckRow({ check, locale }) {
  const percent = check.unit === '%';
  const observed = `${check.observed}${percent ? '%' : ''}`;
  const limit = `${check.limit}${percent ? '%' : ''}`;
  const breached = Number(check.violations || 0) > 0 || check.status === 'at_risk';
  return (
    <li className={`rules-check${breached ? ' breached' : ''}`}>
      <span className="rules-check-label">{locale === 'he' ? check.label_he : check.label_en}</span>
      <span className="rules-check-values" dir="ltr">{observed} / {limit}</span>
      <span className="rules-check-unit">{unitLabel(check.unit, locale)}</span>
      <span className={`rules-check-status${breached ? ' breached' : ''}`}>
        {breached
          ? pageText(locale, 'Needs review', 'דורש בדיקה')
          : pageText(locale, 'Within the limit', 'בתוך המגבלה')}
      </span>
    </li>
  );
}

export default function LicencePage({ locale, session, notify }) {
  const [state, setState] = useState(null);
  const [error, setError] = useState('');
  const [since, setSince] = useState('');

  const load = useCallback((day) => {
    fetchAttestation(day)
      .then((body) => { setState(body); setError(''); })
      .catch((problem) => setError(problem.message));
  }, []);

  useEffect(() => { load(since); }, [load, since]);

  async function change(values, effectiveDate, reason) {
    await recordGuardrailChange(values, effectiveDate, reason);
    notify?.(
      'The licence change is recorded. It takes force on its effective date.',
      'שינוי הרישיון תועד. הוא ייכנס לתוקף בתאריך שנקבע.',
    );
    load(since);
  }

  if (error) {
    return (
      <div className="rules-section">
        <section className="rules-card">
          <h2>{pageText(locale, 'The licence', 'הרישיון')}</h2>
          <p className="rules-inline-error" role="status">
            {pageText(locale, `The compliance verdict is unreachable (${error}).`, `חוות דעת התאימות אינה זמינה (${error}).`)}
          </p>
        </section>
      </div>
    );
  }
  if (!state) {
    return (
      <div className="rules-section">
        <section className="rules-card">
          <p>{pageText(locale, 'Reading the licence', 'קורא את הרישיון')}</p>
        </section>
      </div>
    );
  }

  const compliance = state.compliance || {};
  const licence = state.licence || {};
  const checks = compliance.checks || [];
  const scope = compliance.scope || null;
  const gate = payloadCanEdit(state, session, WALLS.guardrails);
  const scheduled = state.scheduled_changes || [];
  const changes = state.changes_since || [];

  return (
    <div className="rules-section">
      <section className="rules-card">
        <div className="rules-card-head">
          <div>
            <h2>{pageText(locale, 'The verdict', 'חוות הדעת')}</h2>
            <p className="rules-card-lead">
              {pageText(
                locale,
                `${checks.length} checks on the plan of record, judged against the licence in force.`,
                `${checks.length} בדיקות על התוכנית השמורה, מול הרישיון שבתוקף.`,
              )}
            </p>
          </div>
          <span className={`rules-verdict ${compliance.status === 'compliant' ? 'ok' : compliance.status === 'unknown' ? 'unknown' : 'risk'}`}>
            {compliance.status === 'compliant'
              ? pageText(locale, 'Compliant', 'תקין')
              : compliance.status === 'unknown'
                ? pageText(locale, 'Not judged', 'לא נשפט')
                : pageText(locale, 'Needs review', 'דורש בדיקה')}
          </span>
        </div>
        {/* Every figure below is a figure about a population, so the population
            is on the card. The plan of record carries the whole market because
            the retention model is measured against it; what a compliance owner
            signs is the operator's own channel and nobody else's. */}
        {scope && scope.scoped && (
          <p className="rules-scope-line" dir="auto">
            <Tv size={13} aria-hidden="true" />
            <span>
              {pageText(
                locale,
                `Every figure here is ${scope.scope_channel}, the channel this operator owns: ${compliance.graded_breaks} breaks judged, ${scope.competitor_rows_excluded} on ${scope.competitor_channels_excluded} other channels left out.`,
                `כל נתון כאן הוא של ${scope.scope_channel}, הערוץ שבבעלות המפעיל: ${compliance.graded_breaks} ברייקים נשפטו, ${scope.competitor_rows_excluded} בערוצים אחרים (${scope.competitor_channels_excluded}) לא נכללו.`,
              )}
            </span>
          </p>
        )}
        {scope && !scope.scoped && (
          <p className="rules-inline-error" role="status" dir="auto">
            {locale === 'he' ? scope.reason_he : scope.reason_en}
          </p>
        )}
        <ul className="rules-check-list">
          {checks.map((check) => <CheckRow key={check.id} check={check} locale={locale} />)}
        </ul>
        <p className="rules-provenance">
          <ShieldCheck size={14} aria-hidden="true" />
          <span>{licence.profile_name || compliance.profile}</span>
          <span>
            {pageText(
              locale,
              `In force since ${licence.effective_date || compliance.effective_date}`,
              `בתוקף מאז ${licence.effective_date || compliance.effective_date}`,
            )}
          </span>
          {(licence.source_url || compliance.source_url) && (
            <a href={licence.source_url || compliance.source_url} target="_blank" rel="noreferrer">
              {pageText(locale, 'The regulator', 'הרגולטור')}
              <ExternalLink size={12} aria-hidden="true" />
            </a>
          )}
        </p>
        {state.engine_matches_licence === false && (
          <p className="rules-inline-error" role="status">
            {pageText(
              locale,
              'The limits the plan was built against are not the limits recorded here. A recorded change has reached its date and has not been landed, or a limit moved outside this record. Land the licence before you attest.',
              'המגבלות שהתוכנית נבנתה מולן אינן המגבלות הרשומות כאן. שינוי שתועד הגיע לתאריכו ולא הוחל, או שמגבלה זזה מחוץ לתיעוד הזה. החילו את הרישיון לפני האישור.',
            )}
          </p>
        )}
        {/* The payload's disclaimer is authored in English by the compliance
            builder and is part of what Bar 3 protects, so it is still what an
            English reader gets. A Hebrew reader gets the same sentence in the
            page's own language rather than a paragraph they have to translate. */}
        <p className="rules-disclaimer">
          {pageText(
            locale,
            compliance.disclaimer,
            'בסיס שניתן להגדרה. אמתו מול יועץ משפטי עדכני ומול מדיניות המשדר לפני שימוש בייצור.',
          )}
        </p>
      </section>

      <section className="rules-card">
        <div className="rules-card-head">
          <div>
            <h2>{pageText(locale, 'Has anything changed', 'האם משהו השתנה')}</h2>
            <p className="rules-card-lead">
              {pageText(
                locale,
                'Pick the day you last signed. The answer is the evidence you attest with.',
                'בחרו את היום שבו חתמתם לאחרונה. התשובה היא הראיה שאיתה אתם מאשרים.',
              )}
            </p>
          </div>
          <span className="rules-since-field">
            <CalendarClock size={14} aria-hidden="true" />
            <DateField value={since} onChange={setSince} />
          </span>
        </div>
        {state.unchanged ? (
          <p className="rules-attest-ok">
            {state.since_is_whole_log
              ? pageText(
                locale,
                'No regulatory limit has ever been changed in this system. The limits above are the ones it shipped with.',
                'אף מגבלת רגולציה לא שונתה במערכת הזו. המגבלות שלמעלה הן אלה שאיתן היא הגיעה.',
              )
              : pageText(
                locale,
                `No regulatory limit changed on or after ${state.since}. Checked on ${state.checked_on}.`,
                `אף מגבלת רגולציה לא השתנתה מ-${state.since} ואילך. נבדק ב-${state.checked_on}.`,
              )}
          </p>
        ) : (
          <ul className="rules-change-log">
            {changes.map((entry, index) => (
              <li key={`${entry.recorded_at}-${index}`}>
                <span className="rules-change-when">{entry.effective_date}</span>
                <span className="rules-change-what" dir="auto">
                  {Object.entries(entry.values || {}).map(([key, value]) => (
                    <span className="rules-change-limit" key={key}>
                      <span>{limitLabel(key, locale)}</span>
                      <span dir="ltr" aria-label={pairLabel(locale, entry.before?.[key], value)}>
                        {valuePair(entry.before?.[key], value)}
                      </span>
                    </span>
                  ))}
                </span>
                <span className="rules-change-who">{entry.actor}</span>
                <span className="rules-change-why" dir="auto">{entry.reason}</span>
              </li>
            ))}
          </ul>
        )}
        {scheduled.length > 0 && (
          <p className="rules-scheduled" role="status">
            {pageText(
              locale,
              `${scheduled.length} change is recorded for a future date and is not in force yet.`,
              `${scheduled.length} שינוי תועד לתאריך עתידי ואינו בתוקף עדיין.`,
            )}
          </p>
        )}
      </section>

      <LicenceLimits
        locale={locale}
        values={licence.values || {}}
        effectiveDate={licence.effective_date}
        canEdit={gate.canEdit}
        reason={refusalSentence(gate.reason, locale)}
        onChange={change}
      />
    </div>
  );
}
