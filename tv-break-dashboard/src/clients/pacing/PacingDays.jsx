import SourceFileLink from '../SourceFileLink';
import React from 'react';
import { Button } from '../../studio/actions';
import { Code, Figure, Name } from '../../shell/bidi';
import { formatDay, isWeekendDay, weekdayName } from '../../shell/dates';
import { amount, isolate, pick } from './pacing-helpers';
import PacingGoalLine from './PacingGoalLine';

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

// The three words are the delivery ledger's own, quoted rather than re-authored.
// campaigns_delivery.py publishes AIR_STATE_VOCABULARY and says "Scheduled, not
// aired yet" where this file had drifted to "Booked, not aired yet", so one
// product had two words for one state across two of its own destinations. The
// meaning under each word is the ledger's too: aired here means the traffic log
// records those spots and their time has passed, which is not the same claim as
// a delivery feed confirming they ran.
function stateWord(state, locale) {
  if (state === 'aired') return pick(locale, 'Aired', 'שודר');
  if (state === 'scheduled') return pick(locale, 'Scheduled, not aired yet', 'מתוזמן, טרם שודר');
  return pick(locale, 'Unknown', 'לא ידוע');
}

function stateMeaning(state, locale) {
  if (state === 'aired') {
    return pick(locale, 'The traffic log records these spots and their time has passed.', 'יומן השידור רושם את התשדירים האלה והשעה שלהם עברה.');
  }
  if (state === 'scheduled') {
    return pick(locale, 'The traffic log records these spots and their time is still ahead.', 'יומן השידור רושם את התשדירים האלה והשעה שלהם עוד לפנינו.');
  }
  return pick(locale, 'This day is inside the flight and no per-spot source exists for it.', 'היום הזה נמצא בתוך הטיסה ואין עבורו מקור ברמת התשדיר.');
}

// How many spots a booking rule left out of a day, agreeing with its own number.
// Measured on data/campaign_delivery.csv: of the 32 sourced day rows that carry
// one, 21 carry exactly 1, so the majority case read "1 מתוכם הושמטו", a plural
// verb on a single spot, and "1 of them left out" with no verb at all. The
// numeral is isolated in the Hebrew for the same reason every other numeral on
// this surface is: it opens a right-to-left run and its direction is its own.
function droppedSentence(count, locale) {
  const many = Number(count) !== 1;
  if (locale === 'he') {
    return many
      ? `${isolate(count)} מתוכם הושמטו בגלל כלל הזמנה`
      : 'אחד מתוכם הושמט בגלל כלל הזמנה';
  }
  return many
    ? `${count} of them were left out by a booking rule`
    : '1 of them was left out by a booking rule';
}

// What the rule that left those spots out actually capped, and on how many of
// these days it did it.
//
// The ledger names the rule by an engine key and by nothing else, so the drill
// could say that a booking rule removed spots and never what the booking rule
// was. The sentence is composed on the server from the rule's own row, and the
// id itself never reaches the screen: a reader who cannot act on
// DEFAULT_ONE_PER_BREAK is not helped by being shown it.
//
// A rule the rule file does not hold says so and is still counted, because a
// cause this product cannot name is a third state and not an absence.
function ruleLines(days, rules, locale) {
  const counted = new Map();
  days.forEach((day) => {
    const id = String(day.dropped_rule_id || '');
    if (!id || Number(day.spots_dropped_by_rule || 0) <= 0) return;
    counted.set(id, (counted.get(id) || 0) + 1);
  });
  return Array.from(counted.entries()).map(([id, count]) => {
    const block = (rules || {})[id] || null;
    const rule = block ? pick(locale, block.rule_en, block.rule_he) : '';
    const path = block ? pick(locale, block.path_forward_en, block.path_forward_he) : '';
    const where = pick(
      locale,
      `On ${count} of these broadcast days it left spots out of the count.`,
      `ב-${isolate(count)} מימי השידור האלה הוא השמיט תשדירים מהספירה.`,
    );
    // A rule the rule file does not hold is a THIRD STATE and it is counted.
    // The comment above this function has said so from the start and the filter
    // below it dropped exactly those lines, so a cause the product could not
    // name vanished from the drill instead of being reported as unnamed. That is
    // the honest-math law read backwards: unavailable became absent.
    if (!rule) {
      return {
        id,
        rule: pick(
          locale,
          'A booking rule the rule file does not carry.',
          'כלל הזמנה שקובץ הכללים אינו מחזיק.',
        ),
        path: pick(
          locale,
          'The count is real and the cause cannot be named from this tree.',
          'הספירה אמיתית והסיבה אינה ניתנת לשיום מהעץ הזה.',
        ),
        where,
        named: false,
      };
    }
    return { id, rule, path, where, named: true };
  });
}

function figure(day, locale) {
  if (day.air_state === 'unknown') {
    return <span className="pacing-day-unknown">{pick(locale, 'no source', 'אין מקור')}</span>;
  }
  const points = amount(day.rating_points_planned, 'rating_points', locale);
  const money = amount(day.spend_ils, 'ils', locale);
  return (
    <span className="pacing-day-figures">
      <Name>{points}</Name>
      <Name>{money}</Name>
    </span>
  );
}

export default function PacingDays({ drill, line, second, vocabulary, locale, onRetry }) {
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
        <Button type="button" onClick={onRetry}>{pick(locale, 'Try again', 'נסו שוב')}</Button>
      </div>
    );
  }

  const days = drill.days || [];
  const sources = Array.from(new Set(days.map((day) => day.source_file).filter(Boolean)));
  const rules = ruleLines(days, drill.rules, locale);
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
                className={`${day.air_state}${isWeekendDay(day.broadcast_date) ? ' weekend' : ''}`}>
              <th scope="row">
                {/* The day reads dd/mm/yyyy, which is what an Israeli operator
                    reads, and it is read by shell/dates.js rather than here.
                    Measured on the shipped drill: this column printed the
                    payload's raw 2025-04-27 while the Campaigns tab of the same
                    destination printed 27/04/2025 for the same field. */}
                <Figure>{formatDay(day.broadcast_date)}</Figure>
                <small className="pacing-day-weekday">{weekdayName(day.broadcast_date, locale)}</small>
              </th>
              <td title={stateMeaning(day.air_state, locale)}>{stateWord(day.air_state, locale)}</td>
              <td>
                <Figure>{day.spots === null || day.spots === undefined ? '' : day.spots}</Figure>
                {/* How many of them a booking rule left out. The ledger has
                    carried this on every day row all along and no screen read
                    it. Measured: 32 of the 62 sourced day rows in
                    data/campaign_delivery.csv have one, and all three days that
                    price at zero are among them, so the drill printed a money
                    figure of nought with nothing beside it to say why. The
                    ledger names the rule by its engine key only, so the count is
                    stated and the key is not. */}
                {Number(day.spots_dropped_by_rule || 0) > 0 ? (
                  <small className="pacing-day-dropped">
                    {droppedSentence(day.spots_dropped_by_rule, locale)}
                  </small>
                ) : null}
              </td>
              <td>{figure(day, locale)}</td>
            </tr>
          ))}
        </tbody>
        {/* The figure the rows above add up to, taken from the server rather
            than summed here, so the drill closes on the same number the row
            states and a reader never adds seven cells to check that it does. */}
        {line && line.goal !== null && line.goal !== undefined ? (
          <tfoot>
            <tr>
              <th scope="row" colSpan={3}>
                {pick(locale, 'Counted through the day above', 'נספר עד היום שלמעלה')}
              </th>
              <td>
                <Name>{amount(line.counted.through_counted_day, line.unit, locale)}</Name>
              </td>
            </tr>
          </tfoot>
        ) : null}
      </table>

      {/* The campaign's other goal, in the same component the row above states it
          with. This printed one bare pair, counted of goal, with no verdict, no
          reference and no ratio, while the payload carried all three: measured on
          the shipped board, 48 of 56 rows carry a second goal line and on 10 of
          them its verdict differs from the one the row leads with. */}
      <PacingGoalLine line={second} vocabulary={vocabulary} locale={locale}
                      className="pacing-second-line" />

      {rules.map((line) => (
        <p className="pacing-days-rule" key={line.id}>
          <span>{line.rule}</span>
          <span>{line.where}</span>
          {line.path ? <span>{line.path}</span> : null}
        </p>
      ))}

      {sources.length ? (
        <p className="pacing-days-source">
          {pick(locale, 'Read from ', 'נקרא מתוך ')}
          {sources.map((file, index) => (
            <React.Fragment key={file}>
              {index ? ', ' : null}
              <SourceFileLink name={file} locale={locale} />
            </React.Fragment>
          ))}
        </p>
      ) : null}
    </div>
  );
}
