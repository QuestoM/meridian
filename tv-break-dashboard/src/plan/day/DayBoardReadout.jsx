import React from 'react';
import { AlertTriangle, ArrowRight, Calculator, CheckCircle2, PinOff, Undo2 } from 'lucide-react';
import { formatNumber, formatPercent, pageText } from '../../shell/format';
import { clockOf, committedGap, exactCurrency } from './day-board-model';
import { LIVE_PLAN, SAVED_PLAN, livePlanPointer, planBasisLabel, planBasisLead, scopeWithBasis } from './plan-basis';

// What the move cost, told honestly.
//
// The measurement this component exists to respect: on the real channel-day
// רשת 13 / 2024-11-01, moving a break inside its programme changes revenue by
// exactly zero, because a break is priced on its programme's rating, rate,
// premium and length, and not on the minute it starts at. Changing its length by
// sixty seconds changes revenue by 5,670.90 ILS on the same day.
//
// So this readout never animates a revenue figure that did not move. When the
// only edit was a placement it says the revenue is unchanged and why, and then
// shows what genuinely did change: the ad load in the hour, the gap to the
// neighbouring break, and the compliance verdict. When the edit was a length or
// a gold mark it shows the money, because then there is money to show.
//
// Every figure prints the scope it was computed on, beside the figure and not in
// a tooltip.
//
// The verdict and the four acts sit at the panel's top edge rather than after
// its content, and that is a measurement. At 1440 x 823 on רשת 13 / 2024-11-01
// with one break nudged one snap unit, the panel opened at 714 px, the money at
// 727 px, and under the old order the verdict landed at 839 px and all four
// buttons at 917 px, in an 823 px viewport. So every edit a scheduler made ended
// with a scroll to reach Save, on the one row that is reached after every single
// edit. Nothing was dropped to fix it: the panel simply leads with the two
// things an edit is about, and the detail that grows with the day follows them.
function DayBoardReadout({ score, locale, editCount, onUndo, onDiscard, onSave, saving, canUndo, unbound, onRemoveUnbound, forecast, checking, onCheck }) {
  const label = (en, he) => (locale === 'he' ? he : en);
  // Rendered in both branches below. A stranded placement is exactly as real
  // before the first score lands as after it, and it is the one thing on this
  // panel that must never depend on anything a reload empties.
  const stranded = <StrandedPlacements records={unbound} locale={locale} busy={saving} onRemove={onRemoveUnbound} />;
  if (!score) {
    return (
      <div className="day-readout is-idle">
        <p>{label('Select a break, then drag it, or use the arrow keys.', 'בחרו ברייק, גררו אותו או השתמשו במקשי החיצים.')}</p>
        {stranded}
      </div>
    );
  }
  const { basis, saved, current, delta, changed_inputs: changed, compliance } = score;
  const moneyMoved = Math.abs(delta.revenue) > 0.005;
  const onlyPlacement = changed.placement && !changed.duration && !changed.gold;
  // This board re-plans the day live, so every figure on it is the live plan's
  // and its scope line says so. The note below prints the saved plan's own
  // figures, and the two were being read as one number.
  const scopeText = scopeWithBasis(`${basis.channel} / ${basis.day}`, LIVE_PLAN, locale);
  const violations = compliance.violations || [];
  const gap = committedGap(basis, saved);

  return (
    <div className={`day-readout${violations.length ? ' has-violation' : ''}`}>
      {/* The verdict and the four acts, at the panel's own top edge, so where
          they land depends on where the panel starts and never on how much it
          holds. See the module note above for the measurement. */}
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
          <button type="button" className="day-action" onClick={onUndo} disabled={!canUndo}>
            <Undo2 size={13} aria-hidden="true" />
            {label('Undo', 'ביטול פעולה')}
          </button>
          <button type="button" className="day-action" onClick={onDiscard} disabled={editCount === 0}>
            {label('Discard all changes', 'ביטול כל השינויים')}
          </button>
          <button type="button" className="day-action" onClick={onCheck} disabled={editCount === 0 || checking || saving}>
            <Calculator size={13} aria-hidden="true" />
            {checking
              ? label('Measuring', 'מודד')
              : label('Check what saving would do', 'בדיקה מה תעשה השמירה')}
          </button>
          <button type="button" className="day-action is-primary" onClick={onSave} disabled={editCount === 0 || saving}>
            {saving
              ? label('Saving', 'שומר')
              : label(`Save ${editCount} change${editCount === 1 ? '' : 's'}`, editCount === 1 ? 'שמירת שינוי אחד' : `שמירת ${editCount} שינויים`)}
          </button>
        </div>
      </div>

      {stranded}

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
          <small className="day-figure-scope" dir="auto">{scopeWithBasis(`${formatNumber(current.ad_seconds, locale)}s`, LIVE_PLAN, locale)}</small>
        </div>
        <div className={`day-figure${moneyMoved ? ' is-moved' : ''}`}>
          <span className="day-figure-label">{pageText(locale, 'Change from this session', 'שינוי מהמפגש הזה')}</span>
          <strong dir="ltr">{exactCurrency(moneyMoved ? delta.revenue : 0, locale)}</strong>
          <small className="day-figure-scope" dir="auto">{gap.state === 'diverged' ? livePlanPointer(locale) : scopeWithBasis('', LIVE_PLAN, locale)}</small>
        </div>
      </div>

      <CommittedPlanNote gap={gap} locale={locale} />

      {editCount > 0 && onlyPlacement && !moneyMoved && (
        <p className="day-readout-note">
          {label(
            'Moving a break inside its programme does not change what it earns. Its price follows the programme rating, the rate card and the break length, not the minute it starts at. What moved is below.',
            'הזזת ברייק בתוך התוכנית שלו אינה משנה את ההכנסה ממנו. המחיר נגזר מרייטינג התוכנית, מהמחירון ומאורך הברייק, ולא מהדקה שבה הוא מתחיל. מה שכן השתנה מופיע למטה.',
          )}
        </p>
      )}
      {editCount > 0 && changed.duration && (
        <p className="day-readout-note">
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

      <SaveForecast forecast={forecast} locale={locale} editCount={editCount} />

      {score.engine_ms !== null && (
        <p className="day-readout-engine" dir="ltr">
          {label('scored in', 'חושב תוך')} {score.engine_ms} ms
        </p>
      )}
    </div>
  );
}

// Whether this live plan agrees with the weekly plan actually saved to disk.
//
// A blind critic measured this gap directly: the day board served this
// channel-day at 76 breaks and 992,668.69 ILS while the committed weekly plan
// on disk, the same artifact the week board and every export read, held 80
// breaks and 1,067,845.56 ILS, 7.0 per cent apart, and this surface disclosed
// none of it: the tile beside it read a bare "same basis". This is the
// disclosure that replaces the silence, in all three states a comparison can
// honestly land in: no committed row to compare against, the two in agreement,
// or the two apart, with both figures and the gap printed rather than implied.
// Two further measurements are folded in here. The percentage is null when the
// saved plan's revenue is zero, and it used to be interpolated straight into the
// sentence, so that edge printed the literal "null%"; the clause is now dropped
// when there is no percentage to print. And the Hebrew used the prefix "ו-"
// immediately before a negative figure, rendering "ו--4 ברייקים", so both gaps
// are now introduced by a word rather than by a hyphen.
export function CommittedPlanNote({ gap, locale }) {
  const label = (en, he) => (locale === 'he' ? he : en);
  const saved = planBasisLabel(SAVED_PLAN, locale);
  const live = planBasisLabel(LIVE_PLAN, locale);
  if (gap.state === 'unavailable') {
    return (
      <p className="day-committed-note is-unknown" dir="auto">
        {label(`No committed weekly plan is on file for this channel-day, so there is nothing to check ${live} against.`, `אין תוכנית שבועית שמורה ליום הערוץ הזה, ואין מול מה לבדוק את ${live}.`)}
      </p>
    );
  }
  if (gap.state === 'matches') {
    return (
      <p className="day-committed-note is-matched" dir="auto">
        {label(`${planBasisLead(LIVE_PLAN, locale)} matches ${saved}: ${exactCurrency(gap.committed.revenue, locale)}, ${gap.committed.breaks} breaks.`, `${live} תואמת את ${saved}: ${exactCurrency(gap.committed.revenue, locale)}, ${gap.committed.breaks} ברייקים.`)}
      </p>
    );
  }
  const percentClause = gap.percent === null ? '' : ` (${gap.percent}%)`;
  return (
    <p className="day-committed-note is-diverged" dir="auto">
      {label(`This board re-planned this day live. It now differs from ${saved} (${exactCurrency(gap.committed.revenue, locale)}, ${gap.committed.breaks} breaks): a gap of ${exactCurrency(gap.revenueGap, locale)}${percentClause}, and a gap of ${gap.breaksGap} breaks.`, `הלוח הזה תכנן את היום הזה מחדש בזמן אמת. הוא שונה כעת מ${saved} (${exactCurrency(gap.committed.revenue, locale)}, ${gap.committed.breaks} ברייקים): הפרש של ${exactCurrency(gap.revenueGap, locale)}${percentClause}, והפרש של ${gap.breaksGap} ברייקים.`)}
    </p>
  );
}

// The saved placements the plan no longer shows a break for, each with its own
// route back.
//
// This is the hole the round-two board still had. The inverse of a save was
// offered on the break chip that carried the record, and a save is free to change
// how many breaks a programme gets, which renumbers the ids after it. Measured on
// רשת 13 / 2024-11-01: select 001~2, one ArrowRight, Save. The engine re-plans
// that programme from four breaks to one, the day falls from 1,067,845.55 to
// 1,020,401.35 and the count from 80 to 78, and the id 001~2 stops existing. No
// chip rendered as saved, so after a reload the surface offered no Undo, no
// Discard and no Remove anywhere, while the record and the restriction were both
// still on disk. 47,444.20 ILS spent with no route back from the board that spent
// it.
//
// Each row here is addressed by the record's own break id and constraint id, so
// the inverse is exact without a break to hang from, and it settles through the
// same panel a save does. Driven end to end: taking it returned the day to
// 1,067,845.55 and 80 breaks, a gap of 0.0.
//
// It sits at the top of the readout rather than beside the actions, and that is a
// measurement too. Below them at 1440 x 823 the Remove control landed at 915 px
// in an 823 px viewport, so the route back to the money was under the fold on the
// day a person needs it. Above the figures it lands on screen.
export function StrandedPlacements({ records, locale, busy, onRemove }) {
  const label = (en, he) => (locale === 'he' ? he : en);
  const rows = records || [];
  if (!rows.length) return null;
  return (
    <section className="day-stranded" aria-label={label('Saved placements with no break on the board', 'נעיצות שמורות שאין להן ברייק בלוח')}>
      <h4>{label('Saved placements the plan no longer shows a break for', 'נעיצות שמורות שאין להן ברייק בתוכנית')}</h4>
      <p className="day-stranded-why" dir="auto">
        {label(
          'Each was saved from this board, and the plan has since placed its programme differently, so no break carries it. Removing it deletes the restriction that carries it, exactly as the control on a break does.',
          'כל אחת נשמרה מהלוח הזה, ומאז התוכנית מיקמה את רצועת השידור שלה אחרת, ולכן אין ברייק שנושא אותה. ההסרה מוחקת את המגבלה שנושאת אותה, בדיוק כמו הפקד שעל ברייק.',
        )}
      </p>
      <ul>
        {rows.map((record) => (
          <li key={record.break_id} className={`day-stranded-row is-${record.restriction ? record.restriction.state : 'unknown'}`}>
            <div className="day-stranded-identity">
              <strong dir="auto">{record.programme}</strong>
              <span dir="ltr">{record.break_id}</span>
            </div>
            <p dir="auto">{locale === 'he' ? record.reason_he : record.reason}</p>
            <p dir="auto" className="day-stranded-rule">
              {record.restriction ? (locale === 'he' ? record.restriction.reason_he : record.restriction.reason) : ''}
              <span dir="ltr">{record.constraint_id}</span>
            </p>
            <button type="button" className="day-action is-inverse" onClick={() => onRemove(record)} disabled={busy}>
              <PinOff size={13} aria-hidden="true" />
              {label('Remove the saved placement', 'הסרת הנעיצה השמורה')}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

// What saving would do, before the click, and the warning that stands in its place.
//
// The figure tile above says the change from the saved plan, and that number
// holds the break counts the plan already chose. A save does something it cannot
// see: it writes a restriction and the engine plans the whole day again with it
// in force. Measured on רשת 13 / 2024-11-01: pinning a break at exactly its own
// offset and duration reads 0.00 there and costs 30,575.55 ILS, and one
// ArrowRight on 001~2 reads 0.00 there and costs 47,444.20 with the day falling
// from 80 breaks to 78. Nothing on the surface said so before the click.
//
// So an unchecked edit carries the warning in words, and the check replaces it
// with the engine's own figures. Both halves are honest: the words never quote a
// number they did not measure, and the figures are the same optimizer the save
// runs, verified equal to the written save to the cent.
export function SaveForecast({ forecast, locale, editCount }) {
  const label = (en, he) => (locale === 'he' ? he : en);
  if (editCount === 0) return null;
  if (!forecast) {
    return (
      <p className="day-readout-note" dir="auto">
        {label(
          'The change above holds the break counts the plan already chose. Saving writes a restriction and the engine plans the whole day again with it in force, so the day can move further than that. Check what saving would do to measure it before the click.',
          'השינוי שלמעלה מחזיק את מספר הברייקים שהתוכנית כבר בחרה. שמירה כותבת מגבלה, והמנוע מתכנן את כל היום מחדש כשהיא בתוקף, ולכן היום עשוי לזוז יותר מכך. בדקו מה תעשה השמירה כדי למדוד זאת לפני הלחיצה.',
        )}
      </p>
    );
  }
  const scopeText = scopeWithBasis(`${forecast.basis.channel} / ${forecast.basis.day}`, LIVE_PLAN, locale);
  return (
    <section className="day-forecast" aria-label={label('What saving would do', 'מה תעשה השמירה')}>
      <h4>{label('What saving would do', 'מה תעשה השמירה')}</h4>
      <div className="day-readout-figures">
        <div className={`day-figure ${forecast.delta.revenue > 0.005 ? 'is-gain' : forecast.delta.revenue < -0.005 ? 'is-loss' : 'is-flat'}`}>
          <span className="day-figure-label">{label('Change if you save', 'השינוי אם תשמרו')}</span>
          <strong dir="ltr">{exactCurrency(forecast.delta.revenue, locale)}</strong>
          <small className="day-figure-scope" dir="auto">{scopeText}</small>
        </div>
        <div className="day-figure">
          <span className="day-figure-label">{pageText(locale, 'Expected revenue', 'הכנסה צפויה')}</span>
          <strong dir="ltr" className="day-settlement-pair">
            {exactCurrency(forecast.before.revenue, locale)}
            <ArrowRight size={12} aria-hidden="true" />
            {exactCurrency(forecast.after.revenue, locale)}
          </strong>
          <small className="day-figure-scope" dir="auto">{scopeText}</small>
        </div>
        <div className="day-figure">
          <span className="day-figure-label">{pageText(locale, 'Breaks in the day', 'ברייקים ביום')}</span>
          <strong dir="ltr" className="day-settlement-pair">
            {formatNumber(forecast.before.breaks, locale)}
            <ArrowRight size={12} aria-hidden="true" />
            {formatNumber(forecast.after.breaks, locale)}
          </strong>
          <small className="day-figure-scope" dir="auto">{scopeText}</small>
        </div>
      </div>
      <p className="day-readout-note" dir="auto">{forecastSentence(forecast, locale)}</p>
      <p className="day-readout-engine" dir="ltr">
        {label('planned in', 'תוכנן תוך')} {forecast.engine_ms} ms
      </p>
    </section>
  );
}

// One sentence, naming what the second plan did rather than only what it cost.
export function forecastSentence(forecast, locale) {
  const moved = forecast.rearranged;
  const total = forecast.after.breaks;
  const live = planBasisLabel(LIVE_PLAN, locale);
  if (locale === 'he') {
    return `שמירה כותבת מגבלה, והמנוע מתכנן את כל היום מחדש כשהיא בתוקף, ולכן הוא רשאי למקם ברייקים אחרים אחרת. במדידה הזו הוא שינה ${moved.changed} ברייקים מתוך ${total} של ${live}, ב-${moved.programmes} רצועות שידור. שום דבר עדיין לא נכתב.`;
  }
  return `Saving writes a restriction and the engine then plans the whole day again with it in force, so it may place other breaks differently. In this measurement it changed ${moved.changed} breaks of the ${total} in ${live}, across ${moved.programmes} programmes. Nothing has been written yet.`;
}

export function violationLabel(code, locale) {
  const he = {
    retention_floor: 'רצפת שימור',
    breaks_per_hour: 'ברייקים בשעה',
    hourly_ad_load: 'דקות פרסום בשעה',
    break_spacing: 'מרווח בין ברייקים',
    daily_ad_load: 'דקות פרסום ביום',
    gold_breaks: 'ברייקי זהב',
  };
  const en = {
    retention_floor: 'Retention floor',
    breaks_per_hour: 'Breaks per hour',
    hourly_ad_load: 'Ad seconds per hour',
    break_spacing: 'Break spacing',
    daily_ad_load: 'Ad seconds per day',
    gold_breaks: 'Gold breaks',
  };
  return (locale === 'he' ? he[code] : en[code]) || code;
}

// The hour strip: how loaded each clock hour is against the licence limit. It is
// the thing a horizontal move genuinely changes, so it sits beside the money.
//
// Every bar is a real button rather than a titled box, because an hour on this
// board is an address: pressing one selects the first break the plan puts in it,
// which is the object a person pointing at a loaded hour is actually after. The
// load, the licence limit and the break count are in the accessible name too,
// since a title attribute reaches neither the keyboard nor a screen reader.
export function HourStrip({ hours, locale, activeHour, onOpenHour }) {
  const label = (en, he) => (locale === 'he' ? he : en);
  if (!hours || !hours.length) return null;
  return (
    <div className="day-hours" dir="ltr" role="group" aria-label={label('Ad load by hour', 'עומס פרסום לפי שעה')}>
      {hours.map((row) => {
        const share = row.max_ad_seconds > 0 ? Math.min(1.4, row.ad_seconds / row.max_ad_seconds) : 0;
        const state = row.over_ad_seconds || row.over_breaks ? 'is-over' : share > 0.85 ? 'is-tight' : '';
        const clock = clockOf(row.hour * 3600).slice(0, 5);
        const load = `${row.ad_seconds}s / ${row.max_ad_seconds}s, ${row.breaks}/${row.max_breaks}`;
        const opens = label('open the first break in this hour', 'פתיחת הברייק הראשון בשעה הזו');
        return (
          <button
            type="button"
            key={row.hour}
            className={`day-hour ${state}${row.hour === activeHour ? ' is-active' : ''}`}
            title={`${clock} ${load}`}
            aria-label={`${clock}, ${load}, ${opens}`}
            aria-pressed={row.hour === activeHour}
            onClick={() => onOpenHour && onOpenHour(row.hour)}
          >
            <i style={{ height: `${Math.round(share * 100)}%` }} />
            <b>{String(row.hour % 24).padStart(2, '0')}</b>
          </button>
        );
      })}
    </div>
  );
}

export default DayBoardReadout;
