import React from 'react';
import { Clock3, TableProperties } from 'lucide-react';
import { Numeric, formatCurrency, formatMinutes, formatNumber, pageText } from '../../shell/format';
import { DataTable } from '../../shell/primitives';
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

  return (
    <section className="plan-section" aria-labelledby="plan-supply-title">
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

      <div className="plan-figure-row">
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

      <div className="plan-supply-grid">
        <div className="plan-supply-table">
          <div className="plan-section-subhead">
            <h3>{pageText(locale, 'Supply by broadcast strip', 'היצע לפי רצועת שידור')}</h3>
            <span>{formatNumber(dayparts.length, locale)}</span>
          </div>
          <DataTable
            locale={locale}
            fit
            emptyLabel={pageText(locale, 'No supply rows were found for your channel.', 'לא נמצאו שורות היצע לערוץ שלכם.')}
            rows={dayparts}
            columns={[
              { key: 'daypart', label: pageText(locale, 'Broadcast strip', 'רצועת שידור'), render: (row) => engineDaypartLabel(row.daypart, locale) },
              { key: 'spots', label: pageText(locale, 'Spots', 'ספוטים'), render: (row) => formatNumber(row.spots, locale) },
              { key: 'seconds', label: pageText(locale, 'Minutes', 'דקות'), render: (row) => formatMinutes(row.seconds, locale) },
              { key: 'revenue', label: words.expectedRevenue, render: (row) => formatCurrency(row.revenue, locale) },
            ]}
          />
        </div>
        <div className="plan-supply-hours">
          <div className="plan-section-subhead">
            <h3>{pageText(locale, 'Hourly pressure', 'לחץ שעתי')}</h3>
            <span>
              {revenueAvailable
                ? pageText(locale, 'Booked value', 'ערך מוזמן')
                : pageText(locale, 'Booked minutes', 'דקות מוזמנות')}
            </span>
          </div>
          <div className="bar-list chart-ltr" dir="ltr">
            {hours.slice(0, 24).map((row) => (
              <div className="bar-row" key={row.hour_of_day}>
                <span>{String(row.hour_of_day).padStart(2, '0')}:00</span>
                <i style={{ '--bar': Number((revenueAvailable ? row.revenue : row.seconds) || 0) / maxHourValue }} />
                <strong>{revenueAvailable ? formatCurrency(row.revenue, locale) : formatMinutes(row.seconds, locale)}</strong>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

export default SupplyPanel;
