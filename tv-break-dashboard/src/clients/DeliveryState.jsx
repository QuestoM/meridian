import React from 'react';
import { Figure, isolate } from '../shell/bidi';
import { pageText } from '../shell/format';
import { localized, vocabularyLabel } from './clients-money-helpers';
import {
  AIRED,
  COUNTED_EMPTY,
  SCHEDULED,
  UNKNOWN,
  decimals,
  deliverySlice,
  progressReading,
  spotWord,
} from './delivery-helpers';

// The FIGURES half of this destination's delivery display, and the only ways it
// is allowed to print one. The ledger behind them is tri-state, and each of
// these keeps all three states apart on the screen: what aired is real and says
// how much of the flight it was counted over, what is unknown reads as unknown
// with its own days named, and a percent computed over a partial flight is
// labelled a floor rather than presented as a finished figure.
//
// The sentences that say what these figures were counted ON are in
// DeliveryBasisNotes.jsx, and the two ship together on purpose: a figure is
// rendered by DeliveryCell or DeliveryProgress and its basis by DeliveryBasis on
// the same surface, so no count can reach a reader without the instant it was
// taken at, the file it came from and the days nobody has a source for.

// The counted figure, with the word that says what kind of count it is. A floor
// reads as a floor here and nowhere else has to remember to say it.
function countedText(slice, count, locale) {
  const unit = spotWord(count, locale);
  if (slice.isFloor) {
    return pageText(locale, `at least ${count} ${unit}`, `לפחות ${isolate(count)} ${unit}`);
  }
  return pageText(locale, `${count} ${unit}`, `${isolate(count)} ${unit}`);
}

// The counted figure in the unit the flight was BOOKED in, or null when this
// ledger cannot count that unit.
//
// Why this exists: the drawer printed "booked 100 GRP" and, under the word
// delivered, "at least 3 spots". Two units, one comparison, and the comparison
// the labels invite cannot be made. Measured on the shipped store: every one of
// the 51 booked flights carries a GRP goal, so the pairing was never once
// answered in its own currency.
//
// The ledger holds four of the five bookable units — spots, seconds, rating
// points and priced spend — and does not hold impressions. That fifth case
// returns null and says so on the surface rather than quietly answering the
// question that was not asked.
const GOAL_FIGURES = {
  spots: null,  // already the headline; nothing to convert
  seconds: (totals) => totals.seconds,
  grp: (totals) => totals.ratingPoints,
  ils: (totals) => totals.spendIls,
};

function goalFigure(totals, goal, locale) {
  const read = GOAL_FIGURES[String((goal && goal.kind) || '')];
  if (!read) {
    return '';
  }
  const unit = String((goal && goal.unit) || '').trim();
  const value = decimals(read(totals), 2, locale);
  return unit ? `${value} ${unit}` : value;
}

// A counted figure reads as a floor wherever it is printed, in whichever unit,
// because the days behind it are the same partly-sourced days either way.
function flooredText(text, slice, locale) {
  if (!slice.isFloor) {
    return text;
  }
  return pageText(locale, `at least ${text}`, `לפחות ${isolate(text)}`);
}

function daysText(slice, locale) {
  return pageText(
    locale,
    `${slice.sourcedDays} of ${slice.flightDays} flight days counted`,
    `${isolate(slice.sourcedDays)} מתוך ${isolate(slice.flightDays)} ימי טיסה נספרו`,
  );
}

function stateLabel(state, vocabulary, locale) {
  if (state === COUNTED_EMPTY) {
    return pageText(locale, 'Counted, no spot on file', 'נספר, אין תשדיר רשום');
  }
  return vocabularyLabel(vocabulary, state, locale) || pageText(locale, 'Unknown', 'לא ידוע');
}

// One row's delivery. Pass the flight to read that flight's own days, or leave
// the window out to read the whole campaign. The cell never prints a bare
// number: the state leads, the count follows, and the denominator it was counted
// over is on the line under it.
// ``goal`` is the flight's own booked unit as ``{kind, unit}``: pass it and the
// headline answers in that unit, with the spot count kept underneath. Leave it
// out and the cell reads in spots, which is what a surface with no booked goal
// beside it should say.
export function DeliveryCell({ delivery, window = null, vocabulary = [], goal = null, locale }) {
  const slice = deliverySlice(delivery, window);

  if (slice.state === UNKNOWN) {
    return (
      <span className="clients-delivered">
        <span className="clients-unknown">{pageText(locale, 'unknown', 'לא ידוע')}</span>
        {slice.flightDays > 0 ? <small>{daysText(slice, locale)}</small> : null}
      </span>
    );
  }

  const totals = slice.state === SCHEDULED ? slice.scheduled : slice.aired;
  const counted = totals.spots;
  const inGoalUnit = goalFigure(totals, goal, locale);
  const uncounted = String((goal && goal.kind) || '') === 'impressions';
  const ahead = slice.state === AIRED && slice.scheduled.spots > 0;
  const stillToCome = ahead
    ? (goalFigure(slice.scheduled, goal, locale)
      || `${slice.scheduled.spots} ${spotWord(slice.scheduled.spots, locale)}`)
    : '';
  return (
    <span className="clients-delivered">
      <span className={`clients-air-state ${slice.state}`}>{stateLabel(slice.state, vocabulary, locale)}</span>
      <strong className="numeric">
        <Figure>{inGoalUnit
          ? flooredText(inGoalUnit, slice, locale)
          : countedText(slice, counted, locale)}</Figure>
      </strong>
      {inGoalUnit ? <small><Figure>{countedText(slice, counted, locale)}</Figure></small> : null}
      {uncounted ? (
        <small>
          {pageText(
            locale,
            'This ledger counts spots, not impressions, so the booked unit is not answered here.',
            'הספר הזה סופר תשדירים ולא חשיפות, ולכן היחידה שהוזמנה אינה נענית כאן.',
          )}
        </small>
      ) : null}
      <small>{daysText(slice, locale)}</small>
      {/* In the same unit as the figure above it. A cell that answers the
          counted question in GRP and the remaining question in spots makes a
          reader convert between two currencies to read one row. */}
      {stillToCome ? (
        <small>
          {pageText(
            locale,
            `Still to come: ${stillToCome}`,
            `עוד לפנינו: ${isolate(stillToCome)}`,
          )}
        </small>
      ) : null}
    </span>
  );
}


// One goal and how far the counted figures have got against it. The percent is
// never printed alone: it carries the endpoint's own state word, and a floor is
// read out as a floor in the figure itself rather than only in a note. When
// there is no percent the reason stands in its place, because a goal nobody can
// measure against is a state and not a zero.
function Goal({ label, reading, reason, basis, locale }) {
  return (
    <div className={reading.hasPercent ? 'card card-dense card-body clients-progress' : 'card card-dense card-body clients-progress empty'}>
      <dt>{label}</dt>
      <dd>
        {reading.hasPercent ? (
          <>
            <strong className="numeric">
              <Figure>{reading.isFloor
                ? pageText(locale, `at least ${decimals(reading.percent, 2, locale)}%`, `לפחות ${isolate(`${decimals(reading.percent, 2, locale)}%`)}`)
                : `${decimals(reading.percent, 2, locale)}%`}</Figure>
            </strong>
            <span className={`clients-air-state ${reading.state}`}>
              {reading.isFloor
                ? pageText(locale, 'floor, not a total', 'רף תחתון, לא סכום')
                : pageText(locale, 'counted over the whole flight', 'נספר על פני כל הטיסה')}
            </span>
          </>
        ) : (
          <span className="clients-unknown">
            {reading.state === 'unavailable'
              ? pageText(locale, 'not measurable here', 'לא ניתן למדידה כאן')
              : pageText(locale, 'unknown', 'לא ידוע')}
          </span>
        )}
        {reason ? <small className="clients-basis-note">{reason}</small> : null}
        {reading.hasPercent && basis ? <small className="clients-basis-note">{basis}</small> : null}
      </dd>
    </div>
  );
}

export function DeliveryProgress({ delivery, locale }) {
  if (!delivery) {
    return null;
  }
  const rating = progressReading(delivery.rating_progress);
  const budget = progressReading(delivery.budget_progress);
  return (
    <dl className="clients-progress-board">
      <Goal
        label={pageText(locale, 'Against the rating goal', 'מול יעד הרייטינג')}
        reading={rating}
        reason={localized(delivery.rating_progress, 'reason', locale)}
        basis={localized(delivery, 'rating_basis', locale)}
        locale={locale}
      />
      <Goal
        label={pageText(locale, 'Against the budget', 'מול התקציב')}
        reading={budget}
        reason={localized(delivery.budget_progress, 'reason', locale)}
        basis={localized(delivery, 'spend_basis', locale)}
        locale={locale}
      />
    </dl>
  );
}
