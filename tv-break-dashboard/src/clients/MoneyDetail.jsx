import React from 'react';
import { Figure, Code } from '../shell/bidi';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { pageText } from '../shell/format';
import { exactMoney, ledgerBreakKeys, ledgerCampaignKeys, localized } from './clients-money-helpers';
import { isolate } from '../shell/bidi';

// The rows behind one figure. Two levels below the group row: its campaigns,
// and the individual spots that make each of them up, with the break each spot
// sat in as a live chip that re-groups the same ledger by that break.
//
// Every name in here is the address of the rows behind it, including the name at
// the head. The head was the last exception: it printed the client, the agency,
// the campaign or the break as a bare heading, while the agency it named opened
// from two other cells on this same destination and the campaign it named is a
// booked record with its terms and its flights. The head is a control now in
// each of the three groupings whose object this destination holds, and it stays
// a label in the fourth, because a break belongs to the plan surfaces.
//
// The dropped rows travel with it on purpose. The shipped frequency rule removes
// a third of the day, and money that is not there for a stated reason is not the
// same thing as money that is zero, so the rule that removed each spot is
// printed on the row it removed.
//
// Those rows were the last place on this drill where an object named itself and
// opened nothing. A removed spot states the break it would have sat in, and that
// same id is a chip in the table above it, so it is a chip here too wherever the
// ledger holds a row for the break. A break that holds nothing but removed spots
// has no row to open, and its id stays a label rather than becoming a control
// that would land the reader on an empty state.

// What the head of this drill opens, and the words for the control that opens
// it. Two guards, and both are the same one the campaign rows below already use.
// The caller has to have supplied the opener, and the name has to resolve to a
// record: the client record is keyed by the very name the ledger carries, so
// that name is its own address, while an agency record and a booked campaign are
// keyed by an id the ledger does not hold, so those two open only once the index
// the workspace built has resolved them.
function headOpener(field, title, openers) {
  if (field === 'advertiser' && openers.onOpenClient) {
    return {
      open: () => openers.onOpenClient(title),
      en: 'Open the client record',
      he: 'פתחו את כרטיס הלקוח',
    };
  }
  if (field === 'agency' && openers.onOpenAgency && (openers.agencyIds || {})[title]) {
    return {
      open: () => openers.onOpenAgency(openers.agencyIds[title]),
      en: 'Open the agency record',
      he: 'פתחו את כרטיס הסוכנות',
    };
  }
  if (field === 'campaign' && openers.onOpenCampaignRecord && (openers.campaignIds || {})[title]) {
    return {
      open: () => openers.onOpenCampaignRecord(openers.campaignIds[title]),
      en: 'Open the booked campaign',
      he: 'פתחו את הקמפיין שהוזמן',
    };
  }
  return null;
}

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
                <Code className="numeric">{spot.break_id}</Code>
              </button>
            </td>
            <td>{spot.programme}</td>
            <td>{spot.ad}</td>
            <td className="numeric"><Figure>{spot.position === null ? '-' : spot.position}</Figure></td>
            <td className="numeric"><Figure>{spot.duration_seconds}</Figure></td>
            <td className="numeric"><Figure>{spot.planned_tvr}</Figure></td>
            <td className="numeric"><Figure>{exactMoney(spot.gross, locale)}</Figure></td>
            <td className="numeric"><Figure>{spot.rebate_percent}%</Figure></td>
            <td className="numeric"><Figure>{exactMoney(spot.net, locale)}</Figure></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function DroppedRows({ dropped, locale, onOpenBreak, openableBreaks }) {
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
            {openableBreaks.includes(String(row.break_id)) ? (
              <button type="button" className="clients-chip" onClick={() => onOpenBreak(row.break_id)}>
                <Code className="numeric">{row.break_id}</Code>
              </button>
            ) : <Code className="numeric">{row.break_id}</Code>}
            <strong>{row.ad || row.campaign}</strong>
            {row.rule_id ? (
              <span className="clients-rule-id">
                {pageText(locale, `Rule ${row.rule_id}`, `כלל ${isolate(row.rule_id)}`)}
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
  onOpenCampaign,
  openers = {},
}) {
  const keys = new Set(row.spot_keys || []);
  const spots = money.spots.filter((spot) => keys.has(spot.spot_key));
  const droppedKeys = new Set(row.dropped_keys || []);
  const dropped = money.dropped.filter((entry) => droppedKeys.has(entry.spot_key));
  const campaigns = row.campaigns || [];
  const title = String(row[field] || pageText(locale, 'Unnamed', 'ללא שם'));
  const head = headOpener(field, title, openers);
  // Which of these campaign names the board really holds a row for. The campaign
  // rows under a client are the same rows the campaign ranking groups, so on the
  // shipped ledger this is all of them, and a name the ranking does not hold
  // stays a label rather than becoming a control that opens nothing.
  const openable = onOpenCampaign ? ledgerCampaignKeys(money) : [];
  // And which breaks, for the rows a rule removed. The same break id is a chip
  // in the table above and was printed as bare text here, so the break behind a
  // removed spot was the one object on this drill that named itself and opened
  // nothing.
  const openableBreaks = onOpenBreak ? ledgerBreakKeys(money) : [];

  return (
    <article className="clients-detail">
      <header className="clients-detail-head">
        <div>
          <h3>
            {head ? (
              <button type="button" className="clients-link" onClick={head.open}>
                {title}
              </button>
            ) : title}
          </h3>
          <p className="clients-detail-sub">
            <Figure className="numeric">{exactMoney(row.gross, locale)}</Figure>
            <small>{pageText(locale, 'gross', 'ברוטו')}</small>
            <Figure className="numeric">{exactMoney(row.net, locale)}</Figure>
            <small>{pageText(locale, 'net', 'נטו')}</small>
            <Figure className="numeric">{row.spots}</Figure>
            <small>{pageText(locale, 'spots', 'תשדירים')}</small>
            <Figure className="numeric">{(row.share_of_gross * 100).toFixed(2)}%</Figure>
            <small>{pageText(locale, 'of the day', 'מהיום')}</small>
          </p>
        </div>
        <div className="clients-detail-actions">
          {position ? (
            <span className="clients-position">
              <button type="button" onClick={() => onStep(-1)} aria-label={pageText(locale, 'Previous', 'הקודם')}>
                <ChevronRight size={14} aria-hidden="true" />
              </button>
              <Figure className="numeric">{`${position.position} / ${position.total}`}</Figure>
              <button type="button" onClick={() => onStep(1)} aria-label={pageText(locale, 'Next', 'הבא')}>
                <ChevronLeft size={14} aria-hidden="true" />
              </button>
            </span>
          ) : null}
          {head ? (
            <button type="button" className="clients-secondary" onClick={head.open}>
              {pageText(locale, head.en, head.he)}
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
                  <td>
                    {openable.includes(String(campaign.campaign)) ? (
                      <button
                        type="button"
                        className="clients-link"
                        onClick={() => onOpenCampaign(String(campaign.campaign))}
                      >
                        {campaign.campaign}
                      </button>
                    ) : campaign.campaign}
                  </td>
                  <td className="numeric"><Figure>{exactMoney(campaign.gross, locale)}</Figure></td>
                  <td className="numeric"><Figure>{exactMoney(campaign.net, locale)}</Figure></td>
                  <td className="numeric"><Figure>{campaign.spots}</Figure></td>
                  <td className="clients-break-chips">
                    {(campaign.breaks || []).map((breakId) => (
                      <button key={breakId} type="button" className="clients-chip" onClick={() => onOpenBreak(breakId)}>
                        <Code className="numeric">{breakId}</Code>
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
      <DroppedRows
        dropped={dropped}
        locale={locale}
        onOpenBreak={onOpenBreak}
        openableBreaks={openableBreaks}
      />
    </article>
  );
}
