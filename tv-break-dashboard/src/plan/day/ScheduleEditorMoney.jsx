import React from 'react';
import { AlertTriangle, Calculator, CheckCircle2 } from 'lucide-react';
import { formatNumber, formatPercent, pageText } from '../../shell/format';
import { SaveForecast, violationLabel } from './DayBoardReadout';
import { exactCurrency } from './day-board-model';
import './day-readout.css';

// What this day is worth, what the pending edits would do to it, and the check
// that measures the save before it is written.
//
// The editor was a money-blind save surface. Measured by a critic on
// רשת 13 / 2024-11-01: dragging one chip 02:29:00 to 02:22:00 and pressing
// שמור כנעיצה moved the day 1,062,669.88 to 1,037,270.00, and this panel carried
// zero currency figures either side of it. The day board next door had all three
// answers and this surface had none, so the two are now driven from the same
// exports and print the same words: the forecast section and the violation names
// below are literally the board's components, imported rather than copied.
//
// Every figure prints the scope it was computed on, beside the figure and never
// in a tooltip, and nothing here is shown until the engine has answered: before
// the first score lands this panel says so rather than standing a zero in.
function ScheduleEditorMoney({ money, locale, editCount }) {
  const label = (en, he) => (locale === 'he' ? he : en);
  const { score, unaddressable, otherDays } = money;
  const notes = (
    <>
      {unaddressable && unaddressable.length > 0 && (
        <p className="day-readout-note" dir="auto">
          {label(
            `These breaks have no addressable segment in the saved plan, so they are not in the figures above and cannot be saved: ${unaddressable.join(', ')}.`,
            `לברייקים האלה אין מקטע מזוהה בתוכנית השמורה, ולכן הם אינם נכללים בנתונים שלמעלה ולא ניתן לשמור אותם: ${unaddressable.join(', ')}.`,
          )}
        </p>
      )}
      {otherDays && otherDays.length > 0 && (
        <p className="day-readout-note" dir="auto">
          {label(
            `The figures above cover one broadcast day. Edits on ${otherDays.join(', ')} are not counted in them.`,
            `הנתונים שלמעלה מכסים יום שידור אחד. שינויים ב-${otherDays.join(', ')} אינם נכללים בהם.`,
          )}
        </p>
      )}
    </>
  );

  if (!score) {
    return (
      <div className="day-readout is-idle">
        <p dir="auto">{label('Reading what this day is worth.', 'קורא כמה שווה היום הזה.')}</p>
        {notes}
      </div>
    );
  }

  const { basis, current, delta, changed_inputs: changed, compliance } = score;
  const moneyMoved = Math.abs(delta.revenue) > 0.005;
  const onlyPlacement = changed.placement && !changed.duration && !changed.gold;
  const scopeText = `${basis.channel} / ${basis.day}`;
  const violations = compliance.violations || [];

  return (
    <div className={`day-readout${violations.length ? ' has-violation' : ''}`}>
      <div className="day-readout-head">
        {violations.length === 0 ? (
          <span className="day-verdict is-ok">
            <CheckCircle2 size={14} aria-hidden="true" />
            {label(`Compliant on all ${compliance.checks_run} checks`, `תקין בכל ${compliance.checks_run} הבדיקות`)}
          </span>
        ) : (
          <span className="day-verdict is-bad">
            <AlertTriangle size={14} aria-hidden="true" />
            {label(`${violations.length} of ${compliance.checks_run} checks fail`, `${violations.length} מתוך ${compliance.checks_run} בדיקות נכשלות`)}
          </span>
        )}
        <div className="day-readout-actions">
          <button
            type="button"
            className="day-action"
            onClick={money.check}
            disabled={money.moveCount === 0 || money.checking || money.busy}
          >
            <Calculator size={13} aria-hidden="true" />
            {money.checking
              ? label('Measuring', 'מודד')
              : label('Check what saving would do', 'בדיקה מה תעשה השמירה')}
          </button>
        </div>
      </div>

      <div className="day-readout-figures">
        <div className="day-figure">
          <span className="day-figure-label">{pageText(locale, 'Expected revenue', 'הכנסה צפויה')}</span>
          <strong dir="ltr">{exactCurrency(current.revenue, locale)}</strong>
          <small className="day-figure-scope" dir="auto">{scopeText}</small>
        </div>
        <div className="day-figure">
          <span className="day-figure-label">{pageText(locale, 'Retention kept', 'שימור צפייה')}</span>
          <strong dir="ltr">{formatPercent(current.retention * 100, locale)}</strong>
          <small className="day-figure-scope" dir="auto">{scopeText}</small>
        </div>
        <div className="day-figure">
          <span className="day-figure-label">{pageText(locale, 'Breaks in the day', 'ברייקים ביום')}</span>
          <strong dir="ltr">{formatNumber(current.breaks, locale)}</strong>
          <small className="day-figure-scope" dir="ltr">{formatNumber(current.ad_seconds, locale)}s</small>
        </div>
        <div className={`day-figure${moneyMoved ? ' is-moved' : ''}`}>
          <span className="day-figure-label">{pageText(locale, 'Change from the saved plan', 'שינוי מול התוכנית השמורה')}</span>
          <strong dir="ltr">{exactCurrency(moneyMoved ? delta.revenue : 0, locale)}</strong>
          <small className="day-figure-scope">{label('same basis', 'אותו בסיס')}</small>
        </div>
      </div>

      {editCount > 0 && onlyPlacement && !moneyMoved && (
        <p className="day-readout-note" dir="auto">
          {label(
            'Moving a break inside its programme does not change what it earns. Its price follows the programme rating, the rate card and the break length, not the minute it starts at. Saving is a different question, and the check above measures it.',
            'הזזת ברייק בתוך התוכנית שלו אינה משנה את ההכנסה ממנו. המחיר נגזר מרייטינג התוכנית, מהמחירון ומאורך הברייק, ולא מהדקה שבה הוא מתחיל. השמירה היא שאלה אחרת, והבדיקה שלמעלה מודדת אותה.',
          )}
        </p>
      )}
      {editCount > 0 && changed.duration && (
        <p className="day-readout-note" dir="auto">
          {label('Length changed, so the money changed with it.', 'האורך השתנה, ולכן ההכנסה השתנתה איתו.')}
        </p>
      )}

      {violations.length > 0 && (
        <ul className="day-violations" aria-label={label('The checks that fail', 'הבדיקות שנכשלות')}>
          {violations.map((violation, index) => (
            <li key={`${violation.code}-${violation.scope}-${index}`}>
              <strong>{violationLabel(violation.code, locale)}</strong>
              <span dir="auto">{violation.scope}</span>
              <span dir="ltr">{formatNumber(violation.observed, locale)} / {formatNumber(violation.limit, locale)}</span>
            </li>
          ))}
        </ul>
      )}

      {notes}

      <SaveForecast forecast={money.forecast} locale={locale} editCount={money.moveCount} />

      {score.engine_ms !== null && score.engine_ms !== undefined && (
        <p className="day-readout-engine" dir="ltr">
          {label('scored in', 'חושב תוך')} {score.engine_ms} ms
        </p>
      )}
    </div>
  );
}

export default ScheduleEditorMoney;
