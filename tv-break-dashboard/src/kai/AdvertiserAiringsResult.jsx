import React from 'react';
import { Code, Figure, Name } from '../shell/bidi';
import { formatDay, formatDayList } from '../shell/dates';
import { pageText } from '../shell/surface-helpers';
import './advertiser-airings-result.css';

export function advertiserAiringsResult(toolTrace) {
  if (!Array.isArray(toolTrace)) return null;
  const step = toolTrace.find((item) => item && item.tool === 'get_advertiser_airings');
  const result = step && step.result;
  return result && result.kind === 'advertiser_airings' ? result : null;
}

function Metric({ label, value }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd><Figure>{Number(value || 0).toLocaleString()}</Figure></dd>
    </div>
  );
}

function coverageText(coverage, locale) {
  const rows = Number(coverage.selected_rows || 0);
  const authoritative = Number(coverage.authoritative_rows || 0);
  const complete = coverage.complete_for_available_files === true;
  if (locale === 'he') {
    return `${complete ? 'כל' : 'חלק מתוך'} ${rows.toLocaleString('he-IL')} מתוך ${authoritative.toLocaleString('he-IL')} שורות המקור הזמינות`;
  }
  return `${complete ? 'All' : 'Part of'} ${rows.toLocaleString('en-US')} of ${authoritative.toLocaleString('en-US')} available source rows`;
}

export default function AdvertiserAiringsResult({ toolTrace, locale }) {
  const result = advertiserAiringsResult(toolTrace);
  if (!result) return null;
  const summary = result.summary || {};
  const coverage = result.coverage || {};
  const identity = result.identity || {};
  const campaign = Array.isArray(result.campaigns) ? result.campaigns[0] : null;
  const creative = Array.isArray(result.creatives) ? result.creatives[0] : null;
  const airings = Array.isArray(result.airings) ? result.airings : [];
  const days = Array.isArray(coverage.selected_days) ? coverage.selected_days : [];
  const name = identity.shown_name || identity.canonical_name || '';
  return (
    <section className="card asst-airings-card" aria-label={pageText(locale, 'Advertiser airing result', 'תוצאת שידורי המפרסם')}>
      <header>
        <div>
          <span className="asst-airings-kicker">{pageText(locale, 'Sourced airing history', 'היסטוריית שידורים ממקורות')}</span>
          <h3><Name>{name}</Name></h3>
        </div>
        <span className={coverage.complete_for_available_files ? 'asst-airings-state complete' : 'asst-airings-state'}>
          {coverage.complete_for_available_files
            ? pageText(locale, 'Available files fully read', 'הקבצים הזמינים נקראו במלואם')
            : pageText(locale, 'Partial source read', 'קריאת מקור חלקית')}
        </span>
      </header>
      <dl className="asst-airings-metrics">
        <Metric label={pageText(locale, 'Airings', 'שידורים')} value={summary.airings} />
        <Metric label={pageText(locale, 'Seconds', 'שניות')} value={summary.seconds} />
        <Metric label={pageText(locale, 'Broadcast days', 'ימי שידור')} value={summary.broadcast_days} />
        <Metric label={pageText(locale, 'Breaks', 'ברייקים')} value={summary.breaks} />
      </dl>
      <div className="asst-airings-coverage">
        <span>{days.length ? formatDayList(days, locale) : pageText(locale, 'No covered day', 'אין יום מכוסה')}</span>
        <span>{coverageText(coverage, locale)}</span>
        {coverage.complete_through_today === false ? (
          <strong>{pageText(locale, 'This is not lifetime coverage through today.', 'זה אינו כיסוי היסטורי מלא עד היום.')}</strong>
        ) : null}
      </div>
      {campaign || creative ? (
        <dl className="asst-airings-identities">
          {campaign ? <div><dt>{pageText(locale, 'Campaign', 'קמפיין')}</dt><dd><Name>{campaign.campaign}</Name></dd></div> : null}
          {creative ? <div><dt>{pageText(locale, 'Creative', 'קריאייטיב')}</dt><dd><Name>{creative.creative}</Name></dd></div> : null}
          {creative && creative.house_number ? <div><dt>{pageText(locale, 'House number', 'מספר בית')}</dt><dd><Code>{creative.house_number}</Code></dd></div> : null}
        </dl>
      ) : null}
      {airings.length ? (
        <div className="asst-airings-table-wrap">
          <table className="asst-airings-table">
            <thead><tr>
              <th>{pageText(locale, 'Aired', 'שודר')}</th>
              <th>{pageText(locale, 'Programme', 'תוכנית')}</th>
              <th>{pageText(locale, 'Break', 'ברייק')}</th>
              <th>{pageText(locale, 'Position', 'מיקום')}</th>
            </tr></thead>
            <tbody>{airings.map((row, index) => (
              <tr key={`${row.source_file || ''}-${row.source_row || index}`}>
                <td><Figure>{formatDay(row.day)} {row.spot_time || ''}</Figure></td>
                <td><Name>{row.programme || pageText(locale, 'Not available', 'לא זמין')}</Name></td>
                <td><Figure>{row.break_start || pageText(locale, 'Not available', 'לא זמין')}</Figure></td>
                <td><Figure>{row.position_in_break ?? pageText(locale, 'Not available', 'לא זמין')}</Figure></td>
              </tr>
            ))}</tbody>
          </table>
          {result.trace_airings_omitted > 0 ? (
            <p>{pageText(locale, `${result.trace_airings_omitted} more sourced airings are omitted from this compact card.`, `עוד ${result.trace_airings_omitted} שידורים ממקורות אינם מוצגים בכרטיס המקוצר.`)}</p>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
