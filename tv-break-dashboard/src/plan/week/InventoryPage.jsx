import React from 'react';
import { CircleDollarSign, Clock3, ShieldCheck, TableProperties } from 'lucide-react';
import { finiteNumber, formatCurrency, formatMinutes, formatNumber, pageText } from '../../shell/format';
import { riskLabel } from '../../shell/labels';
import { normalizeRows } from '../../shell/plan-model';
import { DataTable, Metric, PageHeader } from '../../shell/primitives';
import { daypartLabel as engineDaypartLabel } from '../../shell/surface-helpers';

export function InventoryPage({ inventory, overview, copy, locale }) {
  // The inventory payload is scoped to the operator's own channel and broken
  // down by broadcast daypart (by_daypart), disclosed via scope_channel. All
  // inventory belongs to that channel, so a market split by channel must not
  // exist here: an older payload that still carries by_channel has no daypart
  // rows and renders an honest empty table, never the channel split.
  const dayparts = normalizeRows(inventory.by_daypart);
  const hours = normalizeRows(inventory.by_hour);
  const scopeChannel = typeof inventory.scope_channel === 'string' ? inventory.scope_channel.trim() : '';
  // The spots source may carry no revenue column; the API then reports
  // revenue: null with revenue_available: false. Say so once instead of
  // leaving the operator to guess why every money figure is a dash.
  const revenueAvailable = inventory.revenue_available !== false;
  const maxHourValue = Math.max(
    ...hours.map((row) => Number((revenueAvailable ? row.revenue : row.seconds) || 0)),
    1,
  );
  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Inventory yield"
        titleHe="תשואת מלאי"
        bodyEn="Check sellable spot supply and hourly demand pressure on your channel before approving a plan."
        bodyHe="בדיקת היצע הספוטים ולחץ הביקוש השעתי בערוץ שלכם לפני אישור תוכנית."
      />
      {scopeChannel && (
        <p className="inventory-scope-line">{pageText(locale, `Your channel's inventory: ${scopeChannel}`, `המלאי של הערוץ שלכם: ${scopeChannel}`)}</p>
      )}
      <section className="metric-strip page-metrics">
        <Metric label={pageText(locale, 'Inventory spots', 'ספוטים במלאי')} value={formatNumber(inventory.summary?.spots, locale)} icon={TableProperties} positive />
        <Metric label={pageText(locale, 'Booked value', 'ערך מוזמן')} value={formatCurrency(inventory.summary?.revenue, locale)} icon={CircleDollarSign} positive />
        <Metric label={pageText(locale, 'Booked minutes', 'דקות מוזמנות')} value={formatMinutes(inventory.summary?.seconds, locale)} icon={Clock3} positive />
        <Metric label={copy.metrics[3]} value={finiteNumber(overview.summary?.risk_score) === null ? '-' : copy.risk[riskLabel(finiteNumber(overview.summary?.risk_score))]} delta={finiteNumber(overview.summary?.risk_score) === null ? '-' : `${finiteNumber(overview.summary?.risk_score)}/100`} icon={ShieldCheck} tone="risk" />
      </section>
      {!revenueAvailable && (
        <p className="data-basis-note">
          {pageText(
            locale,
            'The loaded spots source carries no revenue column, so money figures on this page show a dash. Upload a spots file with revenue to see booked value.',
            'למקור הספוטים שנטען אין עמודת הכנסה, ולכן ערכים כספיים בעמוד זה מוצגים כמקף. העלו קובץ ספוטים עם הכנסה כדי לראות ערך מוזמן.',
          )}
        </p>
      )}
      <div className="page-grid two-one">
        <section className="page-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Inventory by broadcast daypart', 'מלאי לפי רצועת שידור')}</h2>
            <span>{dayparts.length} {pageText(locale, 'dayparts', 'רצועות שידור')}</span>
          </div>
          <DataTable
            locale={locale}
            fit
            emptyLabel={pageText(locale, 'No daypart inventory rows were found.', 'לא נמצאו שורות מלאי לפי רצועת שידור.')}
            rows={dayparts}
            columns={[
              { key: 'daypart', label: pageText(locale, 'Daypart', 'רצועה'), render: (row) => engineDaypartLabel(row.daypart, locale) },
              { key: 'spots', label: pageText(locale, 'Spots', 'ספוטים'), render: (row) => formatNumber(row.spots, locale) },
              { key: 'seconds', label: pageText(locale, 'Minutes', 'דקות'), render: (row) => formatMinutes(row.seconds, locale) },
              { key: 'revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.revenue, locale) },
            ]}
          />
        </section>
        <section className="page-panel">
          <div className="panel-head">
            <h2>{pageText(locale, 'Hourly pressure on your channel', 'לחץ שעתי בערוץ שלכם')}</h2>
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
                <strong>
                  {revenueAvailable ? formatCurrency(row.revenue, locale) : formatMinutes(row.seconds, locale)}
                </strong>
              </div>
            ))}
          </div>
        </section>
      </div>
    </section>
  );
}

export default InventoryPage;
