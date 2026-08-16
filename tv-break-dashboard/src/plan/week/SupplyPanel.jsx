import React, { useEffect, useState } from 'react';
import { Button } from '../../studio/actions';
import { Clock3, TableProperties } from 'lucide-react';
import { Numeric, formatCurrency, formatMinutes, formatNumber, pageText } from '../../shell/format';
import { daypartLabel as engineDaypartLabel } from '../../shell/surface-helpers';
import YieldBlock from './YieldBlock';

// What there is to sell, beside the plan that sells it.
//
// This was a navigation entry of its own called Inventory. It is one question a
// planner asks while planning, so it is a view of Plan rather than a
// destination. The payload is already scoped to the operator's channel by the
// route, and its money column is honestly a dash: the loaded spots source
// carries no revenue column, so the panel names the missing input rather than
// summing zeros into a figure.

export function SupplyPanel({ inventory, locale, words, yieldPerSecond, planScope }) {
  const dayparts = Array.isArray(inventory?.by_daypart) ? inventory.by_daypart : [];
  const hours = Array.isArray(inventory?.by_hour) ? inventory.by_hour : [];
  const scopeChannel = typeof inventory?.scope_channel === 'string' ? inventory.scope_channel.trim() : '';
  const revenueAvailable = inventory?.revenue_available !== false;
  const maxHourValue = Math.max(
    ...hours.map((row) => Number((revenueAvailable ? row.revenue : row.seconds) || 0)),
    1,
  );
  const [selection, setSelection] = useState({ kind: 'daypart', key: dayparts[0]?.daypart || '' });

  useEffect(() => {
    if (selection.kind === 'daypart' && dayparts.some((row) => row.daypart === selection.key)) return;
    if (selection.kind === 'hour' && hours.some((row) => String(row.hour_of_day) === String(selection.key))) return;
    if (dayparts.length) setSelection({ kind: 'daypart', key: dayparts[0].daypart });
    else if (hours.length) setSelection({ kind: 'hour', key: String(hours[0].hour_of_day) });
  }, [dayparts, hours, selection]);

  const selectedRow = selection.kind === 'hour'
    ? hours.find((row) => String(row.hour_of_day) === String(selection.key))
    : dayparts.find((row) => row.daypart === selection.key);
  const selectedLabel = selection.kind === 'hour'
    ? `${String(selectedRow?.hour_of_day ?? 0).padStart(2, '0')}:00`
    : engineDaypartLabel(selectedRow?.daypart, locale);

  return (
    <section className="card plan-section" aria-labelledby="plan-supply-title">
      <div className="plan-section-head">
        <div>
          <h2 id="plan-supply-title">{pageText(locale, 'What there is to sell', 'מה יש למכור')}</h2>
          <p>
            {/* The channel name carries its own direction, so a Hebrew name
                inside an English sentence does not reorder the punctuation. */}
            {scopeChannel
              ? <>{pageText(locale, 'Spot supply and hourly pressure on your channel, ', 'היצע ספוטים ולחץ שעתי בערוץ שלכם, ')}<bdi>{scopeChannel}</bdi>.</>
              : pageText(locale, 'Spot supply and hourly pressure.', 'היצע ספוטים ולחץ שעתי.')}
          </p>
        </div>
      </div>

      <YieldBlock data={yieldPerSecond} locale={locale} words={words} planScope={planScope} />

      <div className="plan-result-ledger plan-supply-summary">
        <div className="plan-figure">
          <span><TableProperties size={13} /> {pageText(locale, 'Spots in supply', 'ספוטים בהיצע')}</span>
          <strong><Numeric>{formatNumber(inventory?.summary?.spots, locale)}</Numeric></strong>
        </div>
        <div className="plan-figure">
          <span><Clock3 size={13} /> {pageText(locale, 'Booked minutes', 'דקות מוזמנות')}</span>
          <strong><Numeric>{formatMinutes(inventory?.summary?.seconds, locale)}</Numeric></strong>
        </div>
        <div className="plan-figure">
          <span>{pageText(locale, 'Dayparts covered', 'רצועות שידור מכוסות')}</span>
          <strong><Numeric>{formatNumber(dayparts.length, locale)}</Numeric></strong>
        </div>
      </div>

      {!revenueAvailable && (
        <p className="plan-note plan-note-amber" role="status">
          {pageText(
            locale,
            'The loaded spots source carries no revenue column, so every money figure on this view is a dash rather than a zero. The path to a figure: upload a spots file that carries revenue, on the Sources workspace.',
            'למקור הספוטים שנטען אין עמודת הכנסה, ולכן כל ערך כספי בתצוגה הזאת הוא מקף ולא אפס. הדרך למספר: העלו קובץ ספוטים שכולל הכנסה, במסך המקורות.',
          )}
        </p>
      )}

      <div className="plan-supply-instrument">
        <div className="plan-supply-strips">
          <div className="plan-section-subhead">
            <h3>{pageText(locale, 'Broadcast strips', 'רצועות שידור')}</h3>
            <span>{formatNumber(dayparts.length, locale)}</span>
          </div>
          <div className="plan-supply-strip-list">
            {dayparts.map((row) => {
              const active = selection.kind === 'daypart' && selection.key === row.daypart;
              return (
                <Button type="button" variant="text" key={row.daypart} className={active ? 'is-active' : undefined} aria-pressed={active} onClick={() => setSelection({ kind: 'daypart', key: row.daypart })}>
                  <span>{engineDaypartLabel(row.daypart, locale)}</span>
                  <strong><Numeric>{formatNumber(row.spots, locale)}</Numeric></strong>
                  <small>{formatMinutes(row.seconds, locale)}</small>
                </Button>
              );
            })}
            {dayparts.length === 0 ? <p>{pageText(locale, 'No supply rows were found for your channel.', 'לא נמצאו שורות היצע לערוץ שלכם.')}</p> : null}
          </div>
        </div>
        <div className="plan-supply-spectrum">
          <div className="plan-section-subhead">
            <h3>{pageText(locale, 'Booked load by hour', 'עומס מוזמן לפי שעה')}</h3>
            <span>{revenueAvailable ? pageText(locale, 'Booked value', 'ערך מוזמן') : pageText(locale, 'Booked minutes', 'דקות מוזמנות')}</span>
          </div>
          <div className="plan-hour-spectrum chart-ltr" dir="ltr">
            {hours.slice(0, 24).map((row) => {
              const active = selection.kind === 'hour' && String(selection.key) === String(row.hour_of_day);
              const ratio = Number((revenueAvailable ? row.revenue : row.seconds) || 0) / maxHourValue;
              const hourLabel = `${String(row.hour_of_day).padStart(2, '0')}:00`;
              const valueLabel = revenueAvailable ? formatCurrency(row.revenue, locale) : formatMinutes(row.seconds, locale);
              return (
                <Button type="button" variant="text" key={row.hour_of_day} className={active ? 'is-active' : undefined} aria-pressed={active} aria-label={`${hourLabel}, ${valueLabel}`} title={`${hourLabel} · ${valueLabel}`} onClick={() => setSelection({ kind: 'hour', key: String(row.hour_of_day) })}>
                  <span>{String(row.hour_of_day).padStart(2, '0')}</span>
                  <i style={{ '--pressure': ratio }} aria-hidden="true" />
                </Button>
              );
            })}
          </div>
        </div>
        <aside className="plan-supply-inspector" aria-live="polite">
          <span className="plan-inspector-eyebrow">{selection.kind === 'hour' ? pageText(locale, 'Selected hour', 'שעה נבחרת') : pageText(locale, 'Selected strip', 'רצועה נבחרת')}</span>
          <h3><bdi>{selectedLabel || '\u2014'}</bdi></h3>
          {selectedRow ? (
            <dl>
              <div><dt>{pageText(locale, 'Spots', 'ספוטים')}</dt><dd><Numeric>{formatNumber(selectedRow.spots, locale)}</Numeric></dd></div>
              <div><dt>{pageText(locale, 'Booked time', 'זמן מוזמן')}</dt><dd><Numeric>{formatMinutes(selectedRow.seconds, locale)}</Numeric></dd></div>
              <div><dt>{words.expectedRevenue}</dt><dd><Numeric>{revenueAvailable ? formatCurrency(selectedRow.revenue, locale) : '\u2014'}</Numeric></dd></div>
            </dl>
          ) : <p>{pageText(locale, 'Select a strip or hour to inspect it.', 'בחרו רצועה או שעה כדי לבדוק אותה.')}</p>}
          {!revenueAvailable ? <small>{pageText(locale, 'The source does not expose revenue, so money remains blank.', 'המקור אינו חושף הכנסה, ולכן הכסף נשאר ריק.')}</small> : null}
        </aside>
      </div>
    </section>
  );
}

export default SupplyPanel;
