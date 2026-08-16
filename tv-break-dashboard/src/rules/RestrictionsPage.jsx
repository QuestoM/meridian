import React, { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, Trash2 } from 'lucide-react';
import { pageText } from '../shell/format';
import ConsequenceDialog, { focusAfterDialogClose } from '../safety/ConsequenceDialog';
import RestrictionComposer from './RestrictionComposer';
import ConstraintBuilder from './ConstraintBuilder';
import { deleteRestriction, detailWords, effectLabel, fetchRestrictions, rulesWrittenSentence, unauthoredSentence } from './rules-lib';
import { formatDay } from '../shell/dates';
import { Pressable } from '../studio/dom-controls';

// A restriction reads as one line. The store's own words are on the record, one
// click away, but nobody has to read them to know what a rule does: the sentence
// is generated from the same record the engine runs, so the list and the plan
// cannot disagree about what is in force.

function RestrictionRow({ record, locale, onRequestDelete }) {
  const expired = record.status === 'expired';
  return (
    <li className={`card rules-restriction${expired ? ' expired' : ''}`}>
      <div className="rules-restriction-main">
        <p className="rules-restriction-sentence">
          {locale === 'he' ? record.sentence_he : record.sentence_en}
        </p>
        <p className="rules-restriction-meta">
          {record.author && (
            <span>{pageText(locale, `Asked by ${record.author}`, `נדרש על ידי ${record.author}`)}</span>
          )}
          {record.reason && <span>{record.reason}</span>}
          {record.starts_on && (
            <span>
              {pageText(locale, `Starts applying on ${formatDay(record.starts_on)}`, `יתחיל לחול ב-${formatDay(record.starts_on)}`)}
            </span>
          )}
          {record.expires_on && (
            <span>
              {expired
                ? pageText(locale, `Stopped applying on ${formatDay(record.expires_on)}`, `הפסיק לחול ב-${formatDay(record.expires_on)}`)
                : pageText(locale, `Stops applying on ${formatDay(record.expires_on)}`, `יפסיק לחול ב-${formatDay(record.expires_on)}`)}
            </span>
          )}
          <span>{rulesWrittenSentence(record.row_count, locale)}</span>
        </p>
      </div>
      <Pressable
        type="button"
        className="rules-icon-button"
        onClick={() => onRequestDelete(record)}
        aria-label={pageText(locale, 'Review removal of this restriction', 'סקירת הסרת ההגבלה')}
      >
        <Trash2 size={14} />
      </Pressable>
    </li>
  );
}

export default function RestrictionsPage({ locale, notify, onGlobalRefresh, onRecompute, recomputeState }) {
  const [state, setState] = useState(null);
  const [error, setError] = useState('');
  const [advanced, setAdvanced] = useState(false);
  const [deleteReview, setDeleteReview] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const restrictionsHeadingRef = React.useRef(null);

  const load = useCallback(() => {
    return fetchRestrictions()
      .then((body) => { setState(body); setError(''); })
      .catch((problem) => setError(detailWords(problem, locale)));
  }, []);

  useEffect(() => { load(); }, [load]);

  async function remove(record) {
    try {
      await deleteRestriction(record.restriction_id);
      notify?.('Restriction removed.', 'ההגבלה הוסרה.');
      await load();
      onGlobalRefresh?.();
      return true;
    } catch (problem) {
      notify?.(`Removing the restriction failed (${detailWords(problem, 'en')}).`, `הסרת ההגבלה נכשלה (${detailWords(problem, 'he')}).`);
      return false;
    }
  }

  async function confirmRemove() {
    if (!deleteReview) return;
    setDeleting(true);
    const removed = await remove(deleteReview);
    setDeleting(false);
    if (removed) {
      setDeleteReview(null);
      focusAfterDialogClose(restrictionsHeadingRef);
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

      <section className="card rules-card">
        <h2 ref={restrictionsHeadingRef} tabIndex={-1}>{pageText(locale, 'Restrictions in force', 'הגבלות בתוקף')}</h2>
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
                onRequestDelete={setDeleteReview}
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
                <li key={row.constraint_id}>
                  <span className="rules-unauthored-effect">{effectLabel(row.effect, locale)}</span>
                  <span>{row.notes || row.scope_value || pageText(locale, 'no description', 'ללא תיאור')}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <ConsequenceDialog
          open={Boolean(deleteReview)}
          locale={locale}
          title={pageText(locale, 'Remove this restriction?', 'להסיר את ההגבלה?')}
          description={pageText(locale, 'Review the authored rule and every stored row that will leave the planning engine.', 'בדקו את הכלל שנכתב ואת כל השורות השמורות שיוסרו ממנוע התכנון.')}
          object={deleteReview ? (
            <span className="consequence-review__object">
              {locale === 'he' ? deleteReview.sentence_he : deleteReview.sentence_en}
              {' · ID '}<bdi>{String(deleteReview.restriction_id)}</bdi>
            </span>
          ) : ''}
          scope={deleteReview ? pageText(
            locale,
            `This restriction and the ${Number(deleteReview.row_count || 0)} stored constraint rows it authored. No other restriction or constraint row changes.`,
            `ההגבלה הזו ו-${Number(deleteReview.row_count || 0)} שורות האילוץ שהיא יצרה. אף הגבלה או שורת אילוץ אחרת לא משתנה.`,
          ) : ''}
          consequence={pageText(locale, 'The restriction stops governing future weekly plan runs. The saved plan is not recomputed by this removal and will need a new run.', 'ההגבלה תפסיק לחול בריצות התכנון השבועיות הבאות. ההסרה אינה מחשבת מחדש את התוכנית השמורה, ויהיה צורך להריץ אותה מחדש.')}
          recovery={pageText(locale, 'A pre-change snapshot is kept on the Restore changes page.', 'תמונת מצב מלפני השינוי נשמרת בעמוד שחזור שינויים.')}
          confirmLabel={pageText(locale, 'Remove restriction', 'הסרת ההגבלה')}
          workingLabel={pageText(locale, 'Removing restriction', 'מסיר את ההגבלה')}
          busy={deleting}
          onCancel={() => setDeleteReview(null)}
          onConfirm={confirmRemove}
        />
      </section>

      <section className="card rules-card rules-advanced">
        <Pressable type="button" className="rules-disclosure" aria-expanded={advanced} onClick={() => setAdvanced(!advanced)}>
          {pageText(locale, 'The condition builder', 'בונה התנאים')}
        </Pressable>
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
