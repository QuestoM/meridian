import React from 'react';
import { Button } from '../../studio/actions';
import { ErrorState, LoadingState } from '../../studio';
import { pageText } from '../../shell/format';

function retryAction(locale, retry) {
  return (
    <Button type="button" variant="outlined" onClick={retry}>
      {pageText(locale, 'Try again', 'ניסיון נוסף')}
    </Button>
  );
}

export function PlanSettingsGate({ state, error, locale, onRetry, children }) {
  if (state === 'ready') return children;
  return (
    <section className="card plan-section" aria-label={pageText(locale, 'Saved objective state', 'מצב המטרה השמורה')}>
      {state === 'loading' ? (
        <LoadingState
          title={pageText(locale, 'Reading the saved objective', 'קורא את המטרה השמורה')}
          description={pageText(locale, 'Plan controls appear only after the saved settings arrive.', 'בקרי התוכנית יופיעו רק לאחר שההגדרות השמורות יגיעו.')}
        />
      ) : (
        <ErrorState
          title={pageText(locale, 'The saved objective is unavailable', 'המטרה השמורה אינה זמינה')}
          description={pageText(locale, `Nothing is inferred from factory values. ${error || ''}`.trim(), `לא מוצגים ערכי ברירת־מחדל במקום ההגדרות השמורות. ${error || ''}`.trim())}
          action={retryAction(locale, onRetry)}
        />
      )}
    </section>
  );
}

export function PlanSectionDataGate({ resource, state, error, locale, onRetry, children }) {
  if (state === 'ready') return children;
  const board = resource === 'schedule';
  const name = pageText(locale, board ? 'week board' : 'supply', board ? 'לוח השבוע' : 'היצע');
  return (
    <section className="card plan-section" aria-label={name}>
      {state === 'loading' || state === 'idle' ? (
        <LoadingState
          title={pageText(locale, `Reading ${name}`, `קורא את ${name}`)}
          description={pageText(locale, 'No business empty state is shown until the source answers.', 'לא מוצג מצב עסקי ריק לפני שמקור הנתונים משיב.')}
        />
      ) : (
        <ErrorState
          title={pageText(locale, `The ${name} data is unavailable`, `נתוני ${name} אינם זמינים`)}
          description={pageText(locale, `The source could not be read. ${error || ''}`.trim(), `לא ניתן היה לקרוא את המקור. ${error || ''}`.trim())}
          action={retryAction(locale, onRetry)}
        />
      )}
    </section>
  );
}

export default PlanSettingsGate;
