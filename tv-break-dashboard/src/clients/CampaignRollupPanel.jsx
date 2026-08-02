import React, { useEffect, useState } from 'react';
import { formatCurrency, formatMinutes, formatNumber, pageText } from '../shell/format';
import { normalizeRows } from '../shell/plan-model';
import { DataTable } from '../shell/primitives';
import { loadRollup } from './clients-api';
import MakeGoodAlerts from './MakeGoodAlerts';

// What aired, as the loaded spots source records it. This is the older of the
// two campaign reads and it stays, because it answers a question the booked
// board cannot: which campaign strings the source data carries at all.
//
// It keeps its honest empty state exactly as measured: when the source has no
// revenue column the rollup ranks by spot count and says so, rather than
// ranking on a fabricated zero.

export default function CampaignRollupPanel({ campaigns, locale, refreshKey }) {
  const [fetched, setFetched] = useState(null);

  useEffect(() => {
    if (campaigns) {
      return undefined;
    }
    let active = true;
    loadRollup()
      .then((payload) => {
        if (active) setFetched(payload);
      })
      .catch(() => {
        if (active) setFetched({ campaigns: [], revenue_available: false });
      });
    return () => { active = false; };
  }, [campaigns, refreshKey]);

  const payload = campaigns || fetched || { campaigns: [] };
  const rows = normalizeRows(payload.campaigns);
  const revenueAvailable = payload.revenue_available !== false;

  return (
    <>
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
          <h2>{pageText(locale, 'Campaigns seen in the source data', 'קמפיינים שנצפו בנתוני המקור')}</h2>
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
    </>
  );
}
