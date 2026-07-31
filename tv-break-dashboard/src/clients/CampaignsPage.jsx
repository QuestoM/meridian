import React from 'react';
import { formatCurrency, formatMinutes, formatNumber, pageText } from '../shell/format';
import { normalizeRows } from '../shell/plan-model';
import { DataTable, PageHeader } from '../shell/primitives';
import MakeGoodAlerts from './MakeGoodAlerts';

export function CampaignsPage({ campaigns, copy, locale, refreshKey }) {
  const rows = normalizeRows(campaigns.campaigns);
  const revenueAvailable = campaigns.revenue_available !== false;
  return (
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Campaign allocation"
        titleHe="הקצאת קמפיינים"
        bodyEn="Track advertiser demand, booked value, channel spread, and the campaigns that constrain optimization."
        bodyHe="מעקב אחר ביקוש מפרסמים, ערך מוזמן, פיזור ערוצים והקמפיינים שמגבילים את האופטימיזציה."
      />
      {!revenueAvailable && (
        <p className="data-basis-note">
          {pageText(
            locale,
            'The loaded spots source carries no revenue column, so campaign revenue shows a dash and campaigns are ranked by spot count.',
            'למקור הספוטים שנטען אין עמודת הכנסה, ולכן הכנסת הקמפיינים מוצגת כמקף והקמפיינים מדורגים לפי מספר ספוטים.',
          )}
        </p>
      )}
      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Advertiser demand', 'ביקוש מפרסמים')}</h2>
          <span>{rows.length} {pageText(locale, 'campaigns', 'קמפיינים')}</span>
        </div>
        <DataTable
          locale={locale}
          emptyLabel={pageText(locale, 'No campaign rows were found.', 'לא נמצאו שורות קמפיינים.')}
          rows={rows}
          columns={[
            { key: 'Campaign', label: pageText(locale, 'Campaign', 'קמפיין') },
            { key: 'advertiser_id', label: pageText(locale, 'Advertiser', 'מפרסם') },
            { key: 'spots', label: pageText(locale, 'Spots', 'ספוטים'), render: (row) => formatNumber(row.spots, locale) },
            { key: 'seconds', label: pageText(locale, 'Minutes', 'דקות'), render: (row) => formatMinutes(row.seconds, locale) },
            { key: 'channels', label: pageText(locale, 'Channels', 'ערוצים'), render: (row) => formatNumber(row.channels, locale) },
            { key: 'revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.revenue, locale) },
            { key: 'last_airing', label: pageText(locale, 'Last airing', 'שידור אחרון') },
          ]}
        />
      </section>
      <MakeGoodAlerts locale={locale} refreshKey={refreshKey} />
    </section>
  );
}

export default CampaignsPage;
