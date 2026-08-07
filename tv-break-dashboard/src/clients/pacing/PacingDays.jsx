import React from 'react';
import { amount, isWeekend, pair, pick, weekday } from './pacing-helpers';

// The broadcast days behind a figure, which is the second level of the drill: a
// figure on the board opens the days it was summed from, and each day names the
// file it was read out of.
//
// The days ride their own read rather than the board payload, so this component
// is handed a read that can be in flight or have failed, and it states which. A
// drill that printed an empty table while its request was still running would
// read as a flight with no days.
//
// A day with no source is a row here, not an absence. It prints the word unknown
// and the sentence the ledger wrote beside it, because a day dropped from the
// table would read as a flight that is shorter than it is.

function stateWord(state, locale) {
  if (state === 'aired') return pick(locale, 'Aired', 'שודר');
  if (state === 'scheduled') return pick(locale, 'Booked, not aired yet', 'מוזמן, טרם שודר');
  return pick(locale, 'Unknown', 'לא ידוע');
}

function figure(day, locale) {
  if (day.air_state === 'unknown') {
    return <span className="pacing-day-unknown">{pick(locale, 'no source', 'אין מקור')}</span>;
  }
  const points = amount(day.rating_points_planned, 'rating_points', locale);
  const money = amount(day.spend_ils, 'ils', locale);
  return (
    <span className="pacing-day-figures">
      <span dir="auto">{points}</span>
      <span dir="auto">{money}</span>
    </span>
  );
}

export default function PacingDays({ drill, second, locale, onRetry }) {
  if (!drill || drill.status === 'loading') {
    return (
      <p className="pacing-loading">
        {pick(locale, 'Reading the broadcast days', 'קורא את ימי השידור')}
      </p>
    );
  }
  if (drill.status === 'failed') {
    return (
      <div className="pacing-failed" role="alert">
        <p>
          {pick(
            locale,
            'The broadcast days behind this row could not be read. What is missing is a failure, not an empty flight.',
            'לא ניתן היה לקרוא את ימי השידור שמאחורי השורה. מה שחסר הוא כשל, לא טיסה ריקה.',
          )}
        </p>
        <button type="button" onClick={onRetry}>{pick(locale, 'Try again', 'נסו שוב')}</button>
      </div>
    );
  }

  const days = drill.days || [];
  const sources = Array.from(new Set(days.map((day) => day.source_file).filter(Boolean)));
  return (
    <div className="pacing-days">
      <table>
        <caption className="pacing-days-caption">
          {pick(
            locale,
            'Every broadcast day of the flight, with what the traffic log holds for it.',
            'כל ימי השידור של הטיסה, ומה שיומן השידור מחזיק עבור כל אחד מהם.',
          )}
        </caption>
        <thead>
          <tr>
            <th scope="col">{pick(locale, 'Broadcast day', 'יום שידור')}</th>
            <th scope="col">{pick(locale, 'State', 'מצב')}</th>
            <th scope="col">{pick(locale, 'Spots', 'תשדירים')}</th>
            <th scope="col">{pick(locale, 'Counted', 'נספר')}</th>
          </tr>
        </thead>
        <tbody>
          {days.map((day) => (
            <tr key={`${day.broadcast_date}-${day.air_state}`}
                className={`${day.air_state}${isWeekend(day.broadcast_date) ? ' weekend' : ''}`}>
              <th scope="row">
                <span dir="ltr">{day.broadcast_date}</span>
                <small className="pacing-day-weekday">{weekday(day.broadcast_date, locale)}</small>
              </th>
              <td>{stateWord(day.air_state, locale)}</td>
              <td dir="ltr">{day.spots === null || day.spots === undefined ? '' : day.spots}</td>
              <td>{figure(day, locale)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {second && second.goal !== null && second.goal !== undefined ? (
        <p className="pacing-second-line">
          {pick(
            locale,
            `The other goal on this campaign: ${pair(second.counted.through_counted_day, second.goal, second.unit, locale)}.`,
            `היעד הנוסף של הקמפיין הזה: ${pair(second.counted.through_counted_day, second.goal, second.unit, locale)}.`,
          )}
        </p>
      ) : null}

      {sources.length ? (
        <p className="pacing-days-source">
          {pick(locale, 'Read from ', 'נקרא מתוך ')}
          <span dir="ltr">{sources.join(', ')}</span>
        </p>
      ) : null}
    </div>
  );
}
