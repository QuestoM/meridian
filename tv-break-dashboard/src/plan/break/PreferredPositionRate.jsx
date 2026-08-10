import React, { useEffect, useState } from 'react';
import { formatNumber, formatPercent, pageText } from '../../shell/format';
import { Figure, Name } from '../../shell/bidi';
import './preferred-position-rate.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

function PreferredPositionRate({ day, locale }) {
  const [state, setState] = useState({ status: 'loading', body: null, error: '' });

  useEffect(() => {
    if (!day) return undefined;
    let active = true;
    setState({ status: 'loading', body: null, error: '' });
    fetch(`${API_BASE}/api/preferred-position-rate?day=${encodeURIComponent(day)}`)
      .then(async (response) => {
        const body = await response.json().catch(() => null);
        if (!response.ok) throw new Error((body && body.detail) || `${response.status} ${response.statusText}`);
        return body || {};
      })
      .then((body) => { if (active) setState({ status: 'ready', body, error: '' }); })
      .catch((error) => { if (active) setState({ status: 'error', body: null, error: error.message }); });
    return () => { active = false; };
  }, [day]);

  const body = state.body || {};
  const he = locale === 'he';
  const label = (en, hebrew) => pageText(locale, en, hebrew);
  const real = body.preferred_state === 'real' && Array.isArray(body.preferred_set);
  const rows = real ? (body.campaigns || []).filter((row) => row.broadcasts > 0) : [];

  return (
    <section className="card card-dense pod-rate">
      <h3>{label('Preferred-position rate', 'שיעור מיקום מועדף')}</h3>
      {state.status === 'loading' && <p>{label('Reading the broadcasts for this day.', 'קורא את השידורים ליום הזה.')}</p>}
      {state.status === 'error' && <p className="pod-warning">{label('The rate could not be read.', 'לא ניתן היה לקרוא את השיעור.')} {state.error}</p>}
      {state.status === 'ready' && !real && (
        <p className="pod-warning">
          {label('Unavailable. No preferred-position set is configured, so no percentage is computed.', 'לא זמין. לא הוגדרה קבוצת מיקומים מועדפים, ולכן לא מחושב אחוז.')}
          {body.preferred_unreadable_reason ? ` ${body.preferred_unreadable_reason}` : ''}
        </p>
      )}
      {state.status === 'ready' && real && (
        <>
          <p>{label('Configured preferred positions', 'מיקומים מועדפים שהוגדרו')}: <Figure>{body.preferred_set.join(', ')}</Figure></p>
          <div className="pod-rate-table" role="table">
            <div className="pod-rate-head" role="row">
              <span role="columnheader">{label('Campaign', 'קמפיין')}</span>
              <span role="columnheader">{label('Broadcasts', 'שידורים')}</span>
              <span role="columnheader">{he ? rows[0]?.agency?.method_label_he : rows[0]?.agency?.method_label_en}</span>
              <span role="columnheader">{he ? rows[0]?.channel?.method_label_he : rows[0]?.channel?.method_label_en}</span>
            </div>
            {rows.map((row) => (
              <div className="pod-rate-row" role="row" key={row.campaign}>
                <Name role="cell">{row.campaign}</Name>
                <Figure role="cell">{formatNumber(row.broadcasts, locale)}</Figure>
                <Figure role="cell">{row.agency.percent === null ? label('unavailable', 'לא זמין') : formatPercent(row.agency.percent, locale)}</Figure>
                <Figure role="cell">{row.channel.percent === null ? label('unavailable', 'לא זמין') : formatPercent(row.channel.percent, locale)}</Figure>
              </div>
            ))}
          </div>
        </>
      )}
      {state.status === 'ready' && (
        <p className="pod-figure-basis">
          <Figure>{formatNumber(body.rows_without_a_campaign_or_position || 0, locale)}</Figure>{' '}
          {label('rows could not enter the denominator because campaign or position was missing.', 'שורות לא נכנסו למכנה מפני שחסר בהן קמפיין או מיקום.')}
        </p>
      )}
    </section>
  );
}

export default PreferredPositionRate;
