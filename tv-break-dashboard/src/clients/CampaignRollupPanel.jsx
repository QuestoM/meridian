import React, { useEffect, useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { exactMoney } from './clients-money-helpers';
import { formatCurrency, formatMinutes, formatNumber, pageText } from '../shell/format';
import { fallbackCampaigns } from '../shell/fallbacks';
import { normalizeRows } from '../shell/plan-model';
import { DataTable } from '../shell/primitives';
import { loadRollup, loadRollupDetail } from './clients-api';
import MakeGoodAlerts from './MakeGoodAlerts';

// What aired, as the loaded spots source records it. This is the older of the
// two campaign reads and it stays, because it answers a question the booked
// board cannot: which campaign strings the source data carries at all.
//
// It keeps its honest empty state exactly as measured: when the source has no
// revenue column the rollup ranks by spot count and says so, rather than
// ranking on a fabricated zero. The advertiser column holds to the same rule:
// when the loaded source has no advertiser column at all, every cell says so
// rather than rendering a blank a reader could mistake for an advertiser that
// really has no name.
//
// A campaign name here used to open nothing, the only dead end on a
// destination where every other name opens the rows behind it. It now opens
// the spot rows GET /api/campaigns/detail holds for that campaign and
// advertiser, inline below the table, in the same shape the money board's own
// drill uses.

function CampaignDrill({ open, locale, onBack }) {
  const [detail, setDetail] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let active = true;
    setDetail(null);
    setFailed(false);
    loadRollupDetail(open.campaign, open.advertiser)
      .then((payload) => { if (active) setDetail(payload); })
      .catch(() => { if (active) setFailed(true); });
    return () => { active = false; };
  }, [open.campaign, open.advertiser]);

  const rows = detail ? normalizeRows(detail.spots) : [];
  const revenueAvailable = !detail || detail.revenue_available !== false;

  return (
    <div className="clients-drill">
      <button type="button" className="clients-back" onClick={onBack}>
        <ArrowLeft size={14} aria-hidden="true" />
        {pageText(locale, 'All campaigns', 'כל הקמפיינים')}
      </button>
      <h3>{open.campaign}</h3>
      {!detail && !failed && (
        <p className="clients-reason">{pageText(locale, 'Loading the spots behind this campaign', 'טוען את התשדירים שמאחורי הקמפיין הזה')}</p>
      )}
      {failed && (
        <p className="clients-reason">{pageText(locale, 'The spot rows could not be read, so none are shown.', 'לא ניתן היה לקרוא את שורות התשדירים, ולכן לא מוצגות שורות.')}</p>
      )}
      {detail && detail.count === 0 && (
        <p className="clients-reason">{pageText(locale, 'No spot rows matched this campaign and advertiser.', 'אף שורת תשדיר לא תאמה לקמפיין ולמפרסם הזה.')}</p>
      )}
      {detail && detail.count > 0 && (
        <>
          <p className="clients-basis-note">
            {detail.count > rows.length
              ? pageText(locale, `Showing the first ${rows.length} of ${detail.count} spots.`, `מוצגים ${rows.length} התשדירים הראשונים מתוך ${detail.count}.`)
              : pageText(locale, `${detail.count} spots.`, `⁦${detail.count}⁩ תשדירים.`)}
          </p>
          <table className="clients-table clients-spots">
            <thead>
              <tr>
                <th scope="col">{pageText(locale, 'Date', 'תאריך')}</th>
                <th scope="col">{pageText(locale, 'Start time', 'שעת התחלה')}</th>
                <th scope="col" className="numeric-col">{pageText(locale, 'Seconds', 'שניות')}</th>
                <th scope="col" className="numeric-col">{pageText(locale, 'Revenue', 'הכנסה')}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={`${row.Date}-${row.start_time}-${index}`}>
                  <td>{row.Date}</td>
                  <td>{row.start_time}</td>
                  <td className="numeric" dir="ltr">{row.Duration}</td>
                  <td className="numeric" dir="ltr">{revenueAvailable ? exactMoney(row.revenue_ils, locale) : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="data-basis-note">
            {pageText(
              locale,
              'Each date and start time is as the loaded spots source records it. No time zone is declared on that source, so which rows fall in a range and how their times read are both on the source clock.',
              'כל תאריך ושעת התחלה הם כפי שמקור הספוטים שנטען רושם אותם. לא מוצהר אזור זמן במקור הזה, ולכן אילו שורות נופלות בטווח וכיצד נקראות השעות שלהן, שניהם לפי שעון המקור.',
            )}
          </p>
          {!revenueAvailable && (
            <p className="data-basis-note">
              {pageText(
                locale,
                'The loaded spots source carries no revenue column, so spot revenue shows a dash.',
                'למקור הספוטים שנטען אין עמודת הכנסה, ולכן הכנסת התשדיר מוצגת כמקף.',
              )}
            </p>
          )}
        </>
      )}
    </div>
  );
}

export default function CampaignRollupPanel({ campaigns, locale, refreshKey }) {
  const [fetched, setFetched] = useState(null);
  const [failed, setFailed] = useState(false);
  const [open, setOpen] = useState(null);

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
    setOpen(null);
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
  const advertiserAvailable = !payload || payload.advertiser_available !== false;
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
      {!advertiserAvailable && (
        <p className="data-basis-note">
          {pageText(
            locale,
            'The loaded spots source carries no advertiser column, so the advertiser is reported not available rather than a blank cell.',
            'למקור הספוטים שנטען אין עמודת מפרסם, ולכן המפרסם מדווח כלא זמין במקום תא ריק.',
          )}
        </p>
      )}
      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Campaigns seen in the source data', 'קמפיינים שנצפו בנתוני המקור')}</h2>
          <span>{countLabel}</span>
        </div>
        {open ? (
          <CampaignDrill open={open} locale={locale} onBack={() => setOpen(null)} />
        ) : (
          <DataTable
            locale={locale}
            emptyLabel={emptyLabel}
            rows={rows}
            columns={[
              {
                key: 'Campaign',
                label: pageText(locale, 'Campaign', 'קמפיין'),
                render: (row) => (
                  <button
                    type="button"
                    className="clients-link"
                    onClick={() => setOpen({ campaign: row.Campaign, advertiser: row.advertiser_id || '' })}
                  >
                    {row.Campaign}
                  </button>
                ),
              },
              {
                key: 'advertiser_id',
                label: pageText(locale, 'Advertiser', 'מפרסם'),
                render: (row) => {
                  if (row.advertiser_id) {
                    return row.advertiser_id;
                  }
                  return advertiserAvailable ? '-' : pageText(locale, 'not available', 'לא זמין');
                },
              },
              { key: 'spots', label: pageText(locale, 'Spots', 'ספוטים'), render: (row) => formatNumber(row.spots, locale) },
              { key: 'seconds', label: pageText(locale, 'Minutes', 'דקות'), render: (row) => formatMinutes(row.seconds, locale) },
              { key: 'revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.revenue, locale) },
              { key: 'last_airing', label: pageText(locale, 'Last airing', 'שידור אחרון') },
            ]}
          />
        )}
      </section>
      <MakeGoodAlerts locale={locale} refreshKey={refreshKey} />
    </>
  );
}
