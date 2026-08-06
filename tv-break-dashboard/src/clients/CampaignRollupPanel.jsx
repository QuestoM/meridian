import React, { useEffect, useState } from 'react';
import { formatCurrency, formatMinutes, formatNumber, pageText } from '../shell/format';
import { fallbackCampaigns } from '../shell/fallbacks';
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
  const [failed, setFailed] = useState(false);

  // The shell hands this panel its own placeholder object while the shared read
  // is in flight, and hands the same object back when that read fails. Identity
  // is the honest test for it: this exact object is not a payload, it is the
  // absence of one, and it holds an empty campaign list that nobody counted. So
  // the placeholder is not treated as data, and the panel reads for itself
  // until a real one arrives.
  const supplied = campaigns && campaigns !== fallbackCampaigns ? campaigns : null;

  useEffect(() => {
    if (supplied) {
      return undefined;
    }
    let active = true;
    setFetched(null);
    setFailed(false);
    loadRollup()
      .then((payload) => {
        if (active) setFetched(payload);
      })
      .catch(() => {
        if (active) setFailed(true);
      });
    return () => { active = false; };
  }, [supplied, refreshKey]);

  // Three states, never two. A read in flight is not an empty result: collapsing
  // it into one printed "0 campaigns" over "no campaign rows were found" for the
  // whole of the read, which is a count nobody measured. So the count is a word
  // until a payload lands, and a failed read says it failed.
  const payload = supplied || fetched;
  const rows = payload ? normalizeRows(payload.campaigns) : [];
  const revenueAvailable = !payload || payload.revenue_available !== false;
  const scope = payload ? payload.scope : null;
  const countLabel = payload
    ? `${rows.length} ${pageText(locale, 'campaigns', 'קמפיינים')}`
    : (failed
      ? pageText(locale, 'not loaded', 'לא נטען')
      : pageText(locale, 'loading', 'בטעינה'));
  const emptyLabel = payload
    ? pageText(locale, 'No campaign rows were found.', 'לא נמצאו שורות קמפיינים.')
    : (failed
      ? pageText(locale, 'The campaign rollup could not be read, so no count is shown rather than a zero.', 'לא ניתן היה לקרוא את ריכוז הקמפיינים, ולכן לא מוצג מספר במקום אפס.')
      : pageText(locale, 'Loading campaigns seen in the source data', 'טוען קמפיינים שנצפו בנתוני המקור'));

  // The scope line, printed the way the money board prints its basis: which
  // channel this rollup was held to, and how many other-channel rows that
  // scope left out, so a reader never mistakes a ranking here for one summed
  // across the whole market.
  const scopeLabel = scope && scope.scope_channel
    ? pageText(
      locale,
      `Scoped to ${scope.scope_channel}. ${scope.competitor_rows_excluded} rows on other channels were excluded.`,
      `בהיקף ${scope.scope_channel}. ${scope.competitor_rows_excluded} שורות מערוצים אחרים הוצאו.`,
    )
    : pageText(
      locale,
      'No operator channel is set, so this rollup could not be scoped and may include other channels.',
      'לא הוגדר ערוץ מפעיל, ולכן ריכוז זה לא ניתן היה להעמיד בהיקף וייתכן שהוא כולל ערוצים אחרים.',
    );

  return (
    <>
      {scope && (
        <p className="data-basis-note">{scopeLabel}</p>
      )}
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
          <span>{countLabel}</span>
        </div>
        <DataTable
          locale={locale}
          emptyLabel={emptyLabel}
          rows={rows}
          columns={[
            { key: 'Campaign', label: pageText(locale, 'Campaign', 'קמפיין') },
            { key: 'advertiser_id', label: pageText(locale, 'Advertiser', 'מפרסם') },
            { key: 'spots', label: pageText(locale, 'Spots', 'ספוטים'), render: (row) => formatNumber(row.spots, locale) },
            { key: 'seconds', label: pageText(locale, 'Minutes', 'דקות'), render: (row) => formatMinutes(row.seconds, locale) },
            { key: 'revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.revenue, locale) },
            { key: 'last_airing', label: pageText(locale, 'Last airing', 'שידור אחרון') },
          ]}
        />
      </section>
      <MakeGoodAlerts locale={locale} refreshKey={refreshKey} />
    </>
  );
}
