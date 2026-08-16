import React from 'react';
import { Button } from '../../studio/actions';
import { Code } from '../../shell/bidi';
import { pageText } from '../../shell/format';
import { requestNavigation } from '../../shell/nav';
import './day-run-safety.css';

export function openOptimizerSources() {
  requestNavigation('Sources', { sources: 'files' });
}

function safetyMessage(safety, locale) {
  if (safety.code === 'settings') {
    return pageText(locale, 'Saved settings could not be verified. No day run was started.', 'לא ניתן לאמת את ההגדרות השמורות. הרצת היום לא התחילה.');
  }
  if (safety.code === 'empty') {
    return safety.inventory?.note?.[locale === 'he' ? 'he' : 'en']
      || pageText(locale, 'The present inventory file yielded no usable placement slots. No day run was started.', 'קובץ המלאי הקיים לא הניב משבצות שיבוץ שמישות. הרצת היום לא התחילה.');
  }
  if (safety.code === 'changed') {
    return pageText(locale, 'Saved settings or optimizer inventory changed after the review. Check the inputs and review the run again.', 'ההגדרות השמורות או מלאי האופטימייזר השתנו לאחר הבדיקה. יש לבדוק את הקלטים ולאשר את ההרצה מחדש.');
  }
  return pageText(locale, `Run inputs could not be verified. ${safety.error || ''}`.trim(), `לא ניתן לאמת את קלטי ההרצה. ${safety.error || ''}`.trim());
}

export default function DayRunSafetyNotice({ safety, locale }) {
  if (!safety || safety.status === 'idle' || safety.status === 'ready') return null;
  if (safety.status === 'checking') {
    return <p className="day-run-safety is-checking" role="status">{pageText(locale, 'Checking saved settings and optimizer inventory.', 'בודק את ההגדרות השמורות ואת מלאי האופטימייזר.')}</p>;
  }
  return (
    <div className="day-run-safety is-blocked" role="alert">
      <p>{safetyMessage(safety, locale)}</p>
      {safety.inventory?.path ? <Code>{safety.inventory.path}</Code> : null}
      <div className="day-run-safety-actions">
        <Button type="button" variant="outlined" onClick={safety.retry}>{pageText(locale, 'Check again', 'בדיקה חוזרת')}</Button>
        <Button type="button" variant="text" onClick={openOptimizerSources}>{pageText(locale, 'Open source files', 'פתיחת קובצי המקור')}</Button>
      </div>
    </div>
  );
}
