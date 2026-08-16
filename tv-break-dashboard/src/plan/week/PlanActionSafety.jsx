import React from 'react';
import { Button } from '../../studio/actions';
import { ErrorState, LoadingState } from '../../studio';
import { Code } from '../../shell/bidi';
import { formatNumber, pageText } from '../../shell/format';
import { requestNavigation } from '../../shell/nav';

export function optimizationBlockedReason(settingsState, inventory, locale) {
  if (settingsState === 'loading') return pageText(locale, 'saved settings are still loading', 'ההגדרות השמורות עדיין נטענות');
  if (settingsState !== 'ready') return pageText(locale, 'saved settings could not be verified', 'לא ניתן לאמת את ההגדרות השמורות');
  if (inventory.status === 'loading') return pageText(locale, 'optimizer inventory is still being checked', 'מלאי האופטימייזר עדיין נבדק');
  if (inventory.code === 'empty') return pageText(locale, 'the inventory file yielded no usable placement slots', 'קובץ המלאי לא הניב משבצות שיבוץ שמישות');
  if (inventory.status !== 'ready') return pageText(locale, 'optimizer inventory readiness could not be verified', 'לא ניתן לאמת את מוכנות מלאי האופטימייזר');
  return '';
}

function SourceAction({ locale, onOpenSources }) {
  const open = onOpenSources || (() => requestNavigation('Sources', { sources: 'files' }));
  return (
    <Button type="button" variant="outlined" onClick={open}>
      {pageText(locale, 'Open source files', 'פתיחת קובצי המקור')}
    </Button>
  );
}

export function PlanActionSafety({ settingsState, settingsError, inventory, locale, onRetrySettings, onRetryInventory, onOpenSources }) {
  const note = inventory.note?.[locale === 'he' ? 'he' : 'en'];
  return (
    <div className="plan-action-safety">
      {settingsState === 'loading' ? (
        <LoadingState title={pageText(locale, 'Checking the saved objective', 'בודק את המטרה השמורה')} />
      ) : settingsState !== 'ready' ? (
        <ErrorState
          title={pageText(locale, 'Run and comparison are locked', 'ההרצה וההשוואה נעולות')}
          description={pageText(locale, `Saved settings were not loaded successfully. ${settingsError || ''}`.trim(), `ההגדרות השמורות לא נטענו בהצלחה. ${settingsError || ''}`.trim())}
          action={<Button type="button" variant="outlined" onClick={onRetrySettings}>{pageText(locale, 'Retry settings', 'טעינה חוזרת של ההגדרות')}</Button>}
        />
      ) : null}

      {inventory.status === 'loading' ? (
        <LoadingState
          title={pageText(locale, 'Checking optimizer inventory', 'בודק את מלאי האופטימייזר')}
          description={<Code>{inventory.path}</Code>}
        />
      ) : inventory.status !== 'ready' ? (
        <ErrorState
          title={pageText(locale, 'Run and comparison are locked', 'ההרצה וההשוואה נעולות')}
          description={note || (inventory.code === 'empty'
            ? pageText(locale, 'The present inventory file yielded no usable placement slots.', 'קובץ המלאי הקיים לא הניב משבצות שיבוץ שמישות.')
            : pageText(locale, `Inventory readiness could not be verified. ${inventory.error || ''}`.trim(), `לא ניתן לאמת את מוכנות המלאי. ${inventory.error || ''}`.trim()))}
          action={(
            <>
              <Button type="button" variant="outlined" onClick={onRetryInventory}>{pageText(locale, 'Check again', 'בדיקה חוזרת')}</Button>
              <SourceAction locale={locale} onOpenSources={onOpenSources} />
            </>
          )}
        >
          <p><Code>{inventory.path}</Code></p>
        </ErrorState>
      ) : (
        <div className="plan-note plan-note-quiet" role="status">
          <span>{inventory.mode === 'identity'
            ? pageText(locale, 'The optional placement inventory is absent; the optimizer will run without inventory weighting.', 'קובץ מלאי השיבוץ האופציונלי חסר; האופטימייזר ירוץ ללא שקלול מלאי.')
            : pageText(locale, `${formatNumber(inventory.slots, locale)} usable placement slots verified.`, `אומתו ${formatNumber(inventory.slots, locale)} משבצות שיבוץ שמישות.`)}</span>
          <Code>{inventory.path}</Code>
          <SourceAction locale={locale} onOpenSources={onOpenSources} />
        </div>
      )}
    </div>
  );
}

export default PlanActionSafety;
