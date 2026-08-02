import React from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { pageText } from '../shell/format';
import { exactMoney, localized } from './clients-money-helpers';

// The rows behind one figure. Two levels below the group row: its campaigns,
// and the individual spots that make each of them up, with the break each spot
// sat in as a live chip that re-groups the same ledger by that break.
//
// The dropped rows travel with it on purpose. The shipped frequency rule removes
// a third of the day, and money that is not there for a stated reason is not the
// same thing as money that is zero, so the rule that removed each spot is
// printed on the row it removed.

function SpotRows({ spots, locale, onOpenBreak }) {
  return (
    <table className="clients-table clients-spots">
      <thead>
        <tr>
          <th scope="col">{pageText(locale, 'Break', 'ברייק')}</th>
          <th scope="col">{pageText(locale, 'Programme', 'תוכנית')}</th>
          <th scope="col">{pageText(locale, 'Ad', 'תשדיר')}</th>
          <th scope="col" className="numeric-col">{pageText(locale, 'Position', 'מיקום')}</th>
          <th scope="col" className="numeric-col">{pageText(locale, 'Seconds', 'שניות')}</th>
          <th scope="col" className="numeric-col">{pageText(locale, 'Rating', 'רייטינג')}</th>
          <th scope="col" className="numeric-col">{pageText(locale, 'Gross', 'ברוטו')}</th>
          <th scope="col" className="numeric-col">{pageText(locale, 'Rebate', 'רבייט')}</th>
          <th scope="col" className="numeric-col">{pageText(locale, 'Net', 'נטו')}</th>
        </tr>
      </thead>
      <tbody>
        {spots.map((spot) => (
          <tr key={spot.spot_key}>
            <td>
              <button type="button" className="clients-chip" onClick={() => onOpenBreak(spot.break_id)}>
                <span className="numeric" dir="ltr">{spot.break_id}</span>
              </button>
            </td>
            <td>{spot.programme}</td>
            <td>{spot.ad}</td>
            <td className="numeric" dir="ltr">{spot.position === null ? '-' : spot.position}</td>
            <td className="numeric" dir="ltr">{spot.duration_seconds}</td>
            <td className="numeric" dir="ltr">{spot.planned_tvr}</td>
            <td className="numeric" dir="ltr">{exactMoney(spot.gross, locale)}</td>
            <td className="numeric" dir="ltr">{spot.rebate_percent}%</td>
            <td className="numeric" dir="ltr">{exactMoney(spot.net, locale)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function DroppedRows({ dropped, locale }) {
  if (!dropped.length) {
    return null;
  }
  return (
    <div className="clients-dropped">
      <h4>{pageText(locale, 'Removed before airing', 'הוסרו לפני השידור')}</h4>
      <p className="clients-basis-note">
        {pageText(
          locale,
          'These spots were in the file and a rule removed them, so their money is not in the figures above.',
          'התשדירים האלה היו בקובץ וכלל הסיר אותם, ולכן הכסף שלהם אינו בסכומים שלמעלה.',
        )}
      </p>
      <ul>
        {dropped.map((row) => (
          <li key={row.spot_key}>
            <span className="numeric" dir="ltr">{row.break_id}</span>
            <strong>{row.ad || row.campaign}</strong>
            {row.rule_id ? (
              <span className="clients-rule-id">
                {pageText(locale, `Rule ${row.rule_id}`, `כלל ⁦${row.rule_id}⁩`)}
              </span>
            ) : null}
            <small className="clients-dropped-why">{localized(row, 'explanation', locale)}</small>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function MoneyDetail({
  money,
  row,
  field,
  locale,
  position,
  onStep,
  onOpenBreak,
  onOpenClient,
}) {
  const keys = new Set(row.spot_keys || []);
  const spots = money.spots.filter((spot) => keys.has(spot.spot_key));
  const droppedKeys = new Set(row.dropped_keys || []);
  const dropped = money.dropped.filter((entry) => droppedKeys.has(entry.spot_key));
  const campaigns = row.campaigns || [];
  const title = String(row[field] || pageText(locale, 'Unnamed', 'ללא שם'));

  return (
    <article className="clients-detail">
      <header className="clients-detail-head">
        <div>
          <h3>{title}</h3>
          <p className="clients-detail-sub">
            <span className="numeric" dir="ltr">{exactMoney(row.gross, locale)}</span>
            <small>{pageText(locale, 'gross', 'ברוטו')}</small>
            <span className="numeric" dir="ltr">{exactMoney(row.net, locale)}</span>
            <small>{pageText(locale, 'net', 'נטו')}</small>
            <span className="numeric" dir="ltr">{row.spots}</span>
            <small>{pageText(locale, 'spots', 'תשדירים')}</small>
            <span className="numeric" dir="ltr">{(row.share_of_gross * 100).toFixed(2)}%</span>
            <small>{pageText(locale, 'of the day', 'מהיום')}</small>
          </p>
        </div>
        <div className="clients-detail-actions">
          {position ? (
            <span className="clients-position">
              <button type="button" onClick={() => onStep(-1)} aria-label={pageText(locale, 'Previous', 'הקודם')}>
                <ChevronRight size={14} aria-hidden="true" />
              </button>
              <span className="numeric" dir="ltr">{`${position.position} / ${position.total}`}</span>
              <button type="button" onClick={() => onStep(1)} aria-label={pageText(locale, 'Next', 'הבא')}>
                <ChevronLeft size={14} aria-hidden="true" />
              </button>
            </span>
          ) : null}
          {field === 'advertiser' && onOpenClient ? (
            <button type="button" className="clients-secondary" onClick={() => onOpenClient(title)}>
              {pageText(locale, 'Open the client record', 'פתחו את כרטיס הלקוח')}
            </button>
          ) : null}
        </div>
      </header>

      {campaigns.length ? (
        <div className="clients-campaign-strip">
          <h4>{pageText(locale, 'Campaigns behind this figure', 'הקמפיינים שמאחורי הסכום')}</h4>
          <table className="clients-table">
            <thead>
              <tr>
                <th scope="col">{pageText(locale, 'Campaign', 'קמפיין')}</th>
                <th scope="col" className="numeric-col">{pageText(locale, 'Gross', 'ברוטו')}</th>
                <th scope="col" className="numeric-col">{pageText(locale, 'Net', 'נטו')}</th>
                <th scope="col" className="numeric-col">{pageText(locale, 'Spots', 'תשדירים')}</th>
                <th scope="col">{pageText(locale, 'Breaks', 'ברייקים')}</th>
              </tr>
            </thead>
            <tbody>
              {campaigns.map((campaign) => (
                <tr key={campaign.campaign}>
                  <td>{campaign.campaign}</td>
                  <td className="numeric" dir="ltr">{exactMoney(campaign.gross, locale)}</td>
                  <td className="numeric" dir="ltr">{exactMoney(campaign.net, locale)}</td>
                  <td className="numeric" dir="ltr">{campaign.spots}</td>
                  <td className="clients-break-chips">
                    {(campaign.breaks || []).map((breakId) => (
                      <button key={breakId} type="button" className="clients-chip" onClick={() => onOpenBreak(breakId)}>
                        <span className="numeric" dir="ltr">{breakId}</span>
                      </button>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <h4>{pageText(locale, 'Every spot behind this figure', 'כל תשדיר שמאחורי הסכום')}</h4>
      <SpotRows spots={spots} locale={locale} onOpenBreak={onOpenBreak} />
      <DroppedRows dropped={dropped} locale={locale} />
    </article>
  );
}
