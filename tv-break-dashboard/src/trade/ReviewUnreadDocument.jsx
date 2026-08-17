import { useCallback, useState } from 'react';
import React from 'react';
import { EmptyState } from '../studio';
import { Button } from '../studio/actions';
import { Name } from '../shell/bidi';
import { pageText } from '../shell/format';
import { loadJob, refusalText, startExtraction } from './trade-api';

// The stage before a review exists: the signed document is attached and nobody
// has read it yet.
//
// This is NOT an error, and it used to render as one - the proposal route
// answers 404 and the screen printed the server's English sentence in a red
// box. A stage that has not happened yet is named, and the action that ends it
// is offered on the same screen.
//
// The reading is a background job on the server, so this hook starts it and
// watches it: a real agreement takes minutes, and a spinner with no words reads
// as a hang, so the elapsed time is stated while it runs.

export function useDocumentReading({ agreementId, documentId, locale, notify, onRead }) {
  const [reading, setReading] = useState('');
  const [failure, setFailure] = useState('');

  const run = useCallback(async () => {
    setFailure('');
    setReading(pageText(locale, 'Starting the reading', 'מתחיל את הקריאה'));
    try {
      const started = await startExtraction(agreementId, documentId);
      for (let tick = 0; tick < 240; tick += 1) {
        await new Promise((resolve) => setTimeout(resolve, 2500));
        const job = await loadJob(started.job_id);
        const status = String(job.status || '');
        if (status === 'done') {
          setReading('');
          notify('The document was read. The proposal is ready for review.',
            'המסמך נקרא. ההצעה מוכנה לסקירה.');
          onRead();
          return;
        }
        if (status === 'failed') {
          setReading('');
          setFailure(String(job.error || pageText(locale, 'The reading failed', 'הקריאה נכשלה')));
          return;
        }
        setReading(pageText(
          locale,
          `Reading the document (${Math.round(tick * 2.5)}s)`,
          `קורא את המסמך (${Math.round(tick * 2.5)} שניות)`,
        ));
      }
      setReading('');
      notify('The reading is still running. Reopen the review in a moment.',
        'הקריאה עדיין מתבצעת. יש לפתוח את הסקירה מחדש בעוד רגע.');
    } catch (error) {
      setReading('');
      setFailure(refusalText(error, locale));
    }
  }, [agreementId, documentId, locale, notify, onRead]);

  return { reading, failure, run };
}

export default function ReviewUnreadDocument({
  document, locale, reading, failure, onRun, onClose, canEdit, editRefusal,
}) {
  return (
    <EmptyState
      title={pageText(locale, 'This document has not been read yet', 'המסמך הזה טרם נקרא')}
      description={pageText(
        locale,
        'The signed document is on file; the engine has not read it into proposed terms yet. Reading a full agreement takes minutes, and every clause it produces still has to pass a person before anything binds.',
        'המסמך החתום מצורף; המנוע עדיין לא קרא אותו למונחים מוצעים. קריאה של הסכם שלם אורכת דקות, וכל סעיף שייצא ממנה עובר אדם לפני שדבר מחייב.',
      )}
      action={(
        <div className="trd-header-actions">
          <Button type="button" variant="outlined" onClick={onClose}>
            {pageText(locale, 'Back to the agreements', 'חזרה לרשימת ההסכמים')}
          </Button>
          <Button
            type="button"
            variant="contained"
            onClick={onRun}
            disabled={Boolean(reading) || !canEdit}
            title={canEdit ? undefined : editRefusal}
          >
            {reading || pageText(locale, 'Read the document now', 'קריאת המסמך עכשיו')}
          </Button>
        </div>
      )}
    >
      {document ? (
        <p className="trd-field-hint">
          <Name>{document.filename}</Name>
          {failure ? ` · ${failure}` : ''}
        </p>
      ) : null}
    </EmptyState>
  );
}
