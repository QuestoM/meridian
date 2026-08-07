import React, { useState } from 'react';
import { Button } from '@mui/material';
import { Lock } from 'lucide-react';
import { pageText } from '../shell/format';
import DateField from '../shell/DateField';
import { detailWords, limitBoundsRefusal, limitLabel } from './rules-lib';

// The four numbers that are the licence. They used to sit beside the revenue
// slider with the same permission, which meant the person accountable for them
// could not tell whether they had moved. Here a change needs a date it takes
// force on and a reason, it is refused before the click when the session may not
// make it, and it never lands silently on today.
//
// The names come from the shared table, so the limit list, the change log and
// the attestation say the same words about the same number.

const LIMITS = [
  { key: 'max_ad_minutes_per_hour', step: 0.5, unitEn: 'minutes', unitHe: 'דקות' },
  { key: 'max_breaks_per_hour', step: 1, unitEn: 'breaks', unitHe: 'ברייקים' },
  { key: 'min_break_spacing_minutes', step: 1, unitEn: 'minutes', unitHe: 'דקות' },
  { key: 'protected_program_max_ad_minutes_per_hour', step: 0.5, unitEn: 'minutes', unitHe: 'דקות' },
];

export default function LicenceLimits({ locale, values, bounds, effectiveDate, canEdit, reason, onChange }) {
  const [draft, setDraft] = useState({});
  const [effective, setEffective] = useState('');
  const [why, setWhy] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const moved = Object.entries(draft).filter(([key, value]) => Number(value) !== Number(values[key]));
  // A value the licence cannot hold is refused here, in this reader's own
  // language, off the same bounds the store validates against. The route that
  // answers a rejected change is frozen and forwards only the English half of
  // the server's refusal, so the one case a person reaches by typing is the one
  // case that must never get there. The save stays shut while any value is out.
  const refusals = moved.map(([key, value]) => limitBoundsRefusal(locale, key, value, bounds)).filter(Boolean);
  const ready = moved.length > 0 && Boolean(effective) && refusals.length === 0;

  async function submit() {
    setSaving(true);
    setError('');
    try {
      await onChange(Object.fromEntries(moved.map(([key, value]) => [key, Number(value)])), effective, why);
      setDraft({});
      setEffective('');
      setWhy('');
    } catch (problem) {
      setError(detailWords(problem, locale));
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="rules-card">
      <div className="rules-card-head">
        <div>
          <h2>{pageText(locale, 'The limits themselves', 'המגבלות עצמן')}</h2>
          <p className="rules-card-lead">
            {pageText(
              locale,
              `These four numbers are the licence. In force since ${effectiveDate}.`,
              `ארבעת המספרים האלה הם הרישיון. בתוקף מאז ${effectiveDate}.`,
            )}
          </p>
        </div>
        {!canEdit && (
          <span className="rules-locked">
            <Lock size={13} aria-hidden="true" />
            <span>{reason}</span>
          </span>
        )}
      </div>

      <ul className="rules-limit-list">
        {LIMITS.map((limit) => {
          const current = values[limit.key];
          const value = draft[limit.key] ?? current ?? '';
          const changed = Number(value) !== Number(current);
          return (
            <li key={limit.key} className={changed ? 'changed' : ''}>
              <label htmlFor={`limit-${limit.key}`}>{limitLabel(limit.key, locale)}</label>
              <input
                id={`limit-${limit.key}`}
                type="number"
                step={limit.step}
                value={value}
                disabled={!canEdit}
                onChange={(event) => setDraft({ ...draft, [limit.key]: event.target.value })}
              />
              <span className="rules-limit-unit">{locale === 'he' ? limit.unitHe : limit.unitEn}</span>
              {changed && (
                <span className="rules-limit-was">
                  {pageText(locale, `was ${current}`, `היה ${current}`)}
                </span>
              )}
            </li>
          );
        })}
      </ul>

      {canEdit && moved.length > 0 && (
        <div className="rules-limit-change">
          <label>
            <span>{pageText(locale, 'Takes force on', 'נכנס לתוקף בתאריך')}</span>
            <DateField value={effective} onChange={setEffective} />
          </label>
          <label>
            <span>{pageText(locale, 'Why', 'סיבה')}</span>
            <input type="text" value={why} onChange={(event) => setWhy(event.target.value)} />
          </label>
          <p className="rules-limit-note">
            {pageText(
              locale,
              'A change dated in the future is recorded now and does not move a number until that day.',
              'שינוי בתאריך עתידי מתועד עכשיו ואינו משנה מספר עד אותו יום.',
            )}
          </p>
          {refusals.map((refusal) => (
            <span className="rules-inline-error" role="status" key={refusal}>{refusal}</span>
          ))}
          <Button className="run-button" type="button" variant="contained" disabled={!ready || saving} onClick={submit}>
            {pageText(locale, 'Record the change', 'תיעוד השינוי')}
          </Button>
          {error && <span className="rules-inline-error" role="status">{error}</span>}
        </div>
      )}
    </section>
  );
}
