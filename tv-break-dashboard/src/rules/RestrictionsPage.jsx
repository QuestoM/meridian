import React, { useCallback, useEffect, useState } from 'react';
import { Button } from '@mui/material';
import { AlertTriangle, Trash2 } from 'lucide-react';
import { pageText } from '../shell/format';
import RestrictionComposer from './RestrictionComposer';
import ConstraintBuilder from './ConstraintBuilder';
import { deleteRestriction, effectLabel, fetchRestrictions, rulesWrittenSentence, unauthoredSentence } from './rules-lib';

// A restriction reads as one line. The store's own words are on the record, one
// click away, but nobody has to read them to know what a rule does: the sentence
// is generated from the same record the engine runs, so the list and the plan
// cannot disagree about what is in force.

function RestrictionRow({ record, locale, onDelete }) {
  const [confirming, setConfirming] = useState(false);
  const expired = record.status === 'expired';
  return (
    <li className={`rules-restriction${expired ? ' expired' : ''}`}>
      <div className="rules-restriction-main">
        <p className="rules-restriction-sentence" dir="auto">
          {locale === 'he' ? record.sentence_he : record.sentence_en}
        </p>
        <p className="rules-restriction-meta">
          {record.author && (
            <span>{pageText(locale, `Asked by ${record.author}`, `נדרש על ידי ${record.author}`)}</span>
          )}
          {record.reason && <span dir="auto">{record.reason}</span>}
          {record.starts_on && (
            <span>
              {pageText(locale, `Starts applying on ${record.starts_on}`, `יתחיל לחול ב-${record.starts_on}`)}
            </span>
          )}
          {record.expires_on && (
            <span>
              {expired
                ? pageText(locale, `Stopped applying on ${record.expires_on}`, `הפסיק לחול ב-${record.expires_on}`)
                : pageText(locale, `Stops applying on ${record.expires_on}`, `יפסיק לחול ב-${record.expires_on}`)}
            </span>
          )}
          <span>{rulesWrittenSentence(record.row_count, locale)}</span>
        </p>
      </div>
      {confirming ? (
        <span className="rules-confirm" role="alertdialog">
          <span>{pageText(locale, 'Remove this restriction from the plan?', 'להסיר את ההגבלה מהתוכנית?')}</span>
          <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => onDelete(record.restriction_id)}>
            {pageText(locale, 'Remove', 'הסרה')}
          </Button>
          <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setConfirming(false)}>
            {pageText(locale, 'Cancel', 'ביטול')}
          </Button>
        </span>
      ) : (
        <button
          type="button"
          className="rules-icon-button"
          onClick={() => setConfirming(true)}
          aria-label={pageText(locale, 'Remove this restriction', 'הסרת ההגבלה')}
        >
          <Trash2 size={14} />
        </button>
      )}
    </li>
  );
}

export default function RestrictionsPage({ locale, notify, onGlobalRefresh, onRecompute, recomputeState }) {
  const [state, setState] = useState(null);
  const [error, setError] = useState('');
  const [advanced, setAdvanced] = useState(false);

  const load = useCallback(() => {
    fetchRestrictions()
      .then((body) => { setState(body); setError(''); })
      .catch((problem) => setError(problem.message));
  }, []);

  useEffect(() => { load(); }, [load]);

  async function remove(restrictionId) {
    try {
      await deleteRestriction(restrictionId);
      notify?.('Restriction removed.', 'ההגבלה הוסרה.');
      load();
      onGlobalRefresh?.();
    } catch (problem) {
      notify?.(`Removing the restriction failed (${problem.message}).`, `הסרת ההגבלה נכשלה (${problem.message}).`);
    }
  }

  const records = state?.restrictions || [];
  const unauthored = state?.unauthored_rows || [];

  return (
    <div className="rules-section">
      <RestrictionComposer
        locale={locale}
        notify={notify}
        onSaved={() => { load(); onGlobalRefresh?.(); }}
      />

      <section className="rules-card">
        <h2>{pageText(locale, 'Restrictions in force', 'הגבלות בתוקף')}</h2>
        {error && (
          <p className="rules-inline-error" role="status">
            {pageText(locale, `The restriction list is unreachable (${error}).`, `רשימת ההגבלות אינה זמינה (${error}).`)}
          </p>
        )}
        {!error && records.length === 0 && (
          <p className="rules-empty">
            {pageText(
              locale,
              'No restrictions yet. The first one you write appears here with who asked for it and when it stops.',
              'אין עדיין הגבלות. הראשונה שתכתבו תופיע כאן, עם מי ביקש אותה ומתי היא נפסקת.',
            )}
          </p>
        )}
        {records.length > 0 && (
          <ul className="rules-restriction-list">
            {records.map((record) => (
              <RestrictionRow
                key={record.restriction_id}
                record={record}
                locale={locale}
                onDelete={remove}
              />
            ))}
          </ul>
        )}

        {unauthored.length > 0 && (
          <div className="rules-unauthored">
            <p>
              <AlertTriangle size={14} aria-hidden="true" />
              {unauthoredSentence(unauthored.length, locale)}
            </p>
            <ul>
              {/* The store's key for what a row does is an engine word, and it
                  used to render raw here while the builder below translated the
                  same value. One table, read by both.

                  Every row, not the first six. The sentence above states how
                  many rows bind the plan with no author, and a panel whose whole
                  reason to exist is that an unreadable rule should not bind the
                  plan cannot then hide some of them behind its own count. */}
              {unauthored.map((row) => (
                <li key={row.constraint_id} dir="auto">
                  <span className="rules-unauthored-effect">{effectLabel(row.effect, locale)}</span>
                  <span>{row.notes || row.scope_value || pageText(locale, 'no description', 'ללא תיאור')}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <section className="rules-card rules-advanced">
        <button type="button" className="rules-disclosure" aria-expanded={advanced} onClick={() => setAdvanced(!advanced)}>
          {pageText(locale, 'The condition builder', 'בונה התנאים')}
        </button>
        <p className="rules-card-lead">
          {pageText(
            locale,
            'For a rule the sentence above cannot say: any combination of programme, genre, daypart, weekday, date and hour, joined with and or or.',
            'לכלל שהמשפט שלמעלה אינו יכול לנסח: כל שילוב של תוכנית, ז׳אנר, רצועת שידור, יום, תאריך ושעה, מחוברים ב-וגם או ב-או.',
          )}
        </p>
        {advanced && (
          <ConstraintBuilder
            locale={locale}
            notify={notify || (() => {})}
            onRecompute={onRecompute}
            recomputeState={recomputeState}
            onGlobalRefresh={() => { load(); onGlobalRefresh?.(); }}
          />
        )}
      </section>
    </div>
  );
}
