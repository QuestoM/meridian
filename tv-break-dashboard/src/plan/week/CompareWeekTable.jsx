import React from 'react';
import { finiteNumber, formatCurrency, pageText } from '../../shell/format';
import { Figure, Name } from '../../shell/bidi';
import { isWeekend, weekdayLabel } from './plan-week-model';

// Which day separates the two scenarios most. A highlighted row is a colour and
// a colour is not a statement, so the day the choice really turns on is also
// said, and it is read off the rows that came back rather than assumed.
function biggestRow(rows) {
  return rows.reduce((winner, row) => {
    const value = Math.abs(Number(row?.delta_revenue_net) || 0);
    return value > 0 && value > Math.abs(Number(winner?.delta_revenue_net) || 0) ? row : winner;
  }, null);
}

function biggestLine(best, locale) {
  if (!best) return null;
  const day = weekdayLabel(best.weekday, locale);
  const amount = formatCurrency(Math.abs(Number(best.delta_revenue_net)), locale);
  const en = `The two scenarios differ most on ${day} ${best.date}, by ${amount} of net.`;
  const he = `ההפרש הגדול ביותר בין שני התרחישים הוא ביום ${day} ${best.date}, ${amount} נטו.`;
  return pageText(locale, en, he);
}

// The day cell as a control. The sentence above the table names the day the
// choice turns on, and the day it names is the plan's own broadcast day, so it
// opens on the week board rather than ending the trail.
function DayOpener({ date, weekday, locale, onOpenDay, children }) {
  if (!onOpenDay || !date) return <>{children}</>;
  return (
    <button
      type="button"
      className="plan-compare-day-open"
      onClick={() => onOpenDay(date)}
      aria-label={pageText(locale, `Open ${date} on the week board`, `פתיחת ${date} בלוח השבוע`)}
      title={pageText(locale, 'Open this broadcast day on the week board', 'פתיחת יום השידור הזה בלוח השבוע')}
      data-weekday={weekday || ''}
    >
      {children}
    </button>
  );
}

// The week the comparison actually ran, one broadcast day per row.
//
// The two cards above this are the week's totals, and a total hides where a
// scenario earned or lost. Measured on the reference data, a retention floor of
// 0.72 against 0.80 changes nothing at all on 2024-11-01 and moves 238,832.52
// on 2024-11-03, so the day rows are the difference between "B is worse" and
// "B is worse on Sunday and Monday, and identical on Friday".
//
// The rows fill in as the server decides each day, so a day that has not been
// computed yet reads as waiting rather than as a zero. Nothing here is
// interpolated: a row prints only figures that arrived for it.

function DayCell({ value, locale }) {
  const number = finiteNumber(value);
  if (number === null) return <td className="numeric"><Figure>{pageText(locale, 'waiting', 'ממתין')}</Figure></td>;
  return <td className="numeric"><Figure>{formatCurrency(number, locale)}</Figure></td>;
}

export function CompareWeekTable({ locale, dates, days, running, elapsedMs, biggestDate, onOpenDay }) {
  const rows = Array.isArray(dates) && dates.length
    ? dates.map((date) => days.find((day) => day.date === date) || { date, pending: true })
    : days;
  if (!rows.length) return null;
  const done = days.length;
  const total = Array.isArray(dates) && dates.length ? dates.length : days.length;
  const seconds = finiteNumber(elapsedMs) === null ? null : (Number(elapsedMs) / 1000).toFixed(1);
  const turnRow = biggestRow(days);
  const turnLine = biggestLine(turnRow, locale);

  return (
    <div className="plan-compare-days">
      <div className="plan-compare-days-head">
        <h3>{pageText(locale, 'Every day of the week compared', 'כל ימי השבוע, אחד אחד')}</h3>
        {running ? (
          <p className="plan-compare-progress" role="status" aria-live="polite">
            {pageText(locale, `Day ${done} of ${total}`, `יום ${done} מתוך ${total}`)}
            {seconds === null ? null : <Figure className="numeric">{pageText(locale, `, ${seconds} s so far`, `, ${seconds} שניות עד כה`)}</Figure>}
          </p>
        ) : null}
      </div>
      <table className="plan-compare-table">
        <thead>
          <tr>
            <th scope="col">{pageText(locale, 'Broadcast day', 'יום שידור')}</th>
            <th scope="col">{pageText(locale, 'Scenario A, net', 'תרחיש A, נטו')}</th>
            <th scope="col">{pageText(locale, 'Scenario B, net', 'תרחיש B, נטו')}</th>
            <th scope="col">{pageText(locale, 'Difference', 'הפרש')}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const delta = finiteNumber(row.delta_revenue_net);
            const tone = delta === null || delta === 0 ? '' : delta > 0 ? ' up' : ' down';
            const weekend = isWeekend(row.weekday);
            const reason = row.a?.available === false ? row.a.reason : null;
            return (
              <tr key={row.date} className={`${weekend ? 'is-weekend' : ''}${row.date === biggestDate ? ' is-biggest' : ''}`}>
                <th scope="row">
                  <DayOpener date={row.date} weekday={row.weekday} locale={locale} onOpenDay={onOpenDay}>
                    <Figure className="numeric">{row.date}</Figure>
                    <small>{weekdayLabel(row.weekday, locale)}</small>
                  </DayOpener>
                  {reason ? <small className="plan-compare-reason"><Name>{reason}</Name></small> : null}
                </th>
                <DayCell value={row.a?.revenue_net} locale={locale} />
                <DayCell value={row.b?.revenue_net} locale={locale} />
                <td className={`numeric${tone}`}>
                  <Figure>
                    {delta === null
                      ? pageText(locale, 'waiting', 'ממתין')
                      : `${delta > 0 ? '+' : ''}${formatCurrency(delta, locale)}`}
                  </Figure>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {!running && turnLine ? (
        <p className="plan-compare-turn">
          <DayOpener date={turnRow.date} weekday={turnRow.weekday} locale={locale} onOpenDay={onOpenDay}>
            <span>{turnLine}</span>
          </DayOpener>
        </p>
      ) : null}
      <p className="plan-compare-legend">
        {onOpenDay
          ? pageText(
            locale,
            'Net is expected revenue minus retention cost, for that broadcast day. Friday and Saturday are marked as the weekend. Open a broadcast day to see it on the week board.',
            'נטו הוא הכנסה צפויה בניכוי עלות שימור, ליום השידור הזה. שישי ושבת מסומנים כסוף השבוע. פתחו יום שידור כדי לראות אותו בלוח השבוע.',
          )
          : pageText(
            locale,
            'Net is expected revenue minus retention cost, for that broadcast day. Friday and Saturday are marked as the weekend.',
            'נטו הוא הכנסה צפויה בניכוי עלות שימור, ליום השידור הזה. שישי ושבת מסומנים כסוף השבוע.',
          )}
      </p>
    </div>
  );
}

export default CompareWeekTable;
