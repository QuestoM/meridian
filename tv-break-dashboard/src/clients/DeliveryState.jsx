import React from 'react';
import { pageText } from '../shell/format';
import { localized, vocabularyLabel } from './clients-money-helpers';
import {
  AIRED,
  COUNTED_EMPTY,
  SCHEDULED,
  UNKNOWN,
  deliverySlice,
  droppedRulesOf,
  progressReading,
  sourceFilesOf,
} from './delivery-helpers';

// The three ways this destination is allowed to print delivery, and the only
// three. The ledger behind them is tri-state, and each of these keeps all three
// states apart on the screen: what aired is real and says how much of the flight
// it was counted over, what is unknown reads as unknown with its own days named,
// and a percent computed over a partial flight is labelled a floor rather than
// presented as a finished figure.
//
// They travel together on purpose. A figure is rendered by DeliveryCell or
// DeliveryProgress, and the basis that figure was counted on is rendered by
// DeliveryBasis on the same surface, so no count can reach a reader without the
// instant it was taken at, the file it came from and the days nobody has a
// source for.

function decimals(value, places, locale) {
  return new Intl.NumberFormat(locale === 'he' ? 'he-IL' : 'en-US', {
    maximumFractionDigits: places,
    minimumFractionDigits: 0,
  }).format(Number(value));
}

function spotWord(count, locale) {
  return pageText(locale, count === 1 ? 'spot' : 'spots', count === 1 ? 'תשדיר' : 'תשדירים');
}

// The counted figure, with the word that says what kind of count it is. A floor
// reads as a floor here and nowhere else has to remember to say it.
function countedText(slice, count, locale) {
  const unit = spotWord(count, locale);
  if (slice.isFloor) {
    return pageText(locale, `at least ${count} ${unit}`, `לפחות ⁦${count}⁩ ${unit}`);
  }
  return pageText(locale, `${count} ${unit}`, `⁦${count}⁩ ${unit}`);
}

function daysText(slice, locale) {
  return pageText(
    locale,
    `${slice.sourcedDays} of ${slice.flightDays} flight days counted`,
    `⁦${slice.sourcedDays}⁩ מתוך ⁦${slice.flightDays}⁩ ימי טיסה נספרו`,
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
export function DeliveryCell({ delivery, window = null, vocabulary = [], locale }) {
  const slice = deliverySlice(delivery, window);

  if (slice.state === UNKNOWN) {
    return (
      <span className="clients-delivered">
        <span className="clients-unknown">{pageText(locale, 'unknown', 'לא ידוע')}</span>
        {slice.flightDays > 0 ? <small>{daysText(slice, locale)}</small> : null}
      </span>
    );
  }

  const counted = slice.state === SCHEDULED ? slice.scheduled.spots : slice.aired.spots;
  const stillToCome = slice.state === AIRED && slice.scheduled.spots > 0 ? slice.scheduled.spots : 0;
  return (
    <span className="clients-delivered">
      <span className={`clients-air-state ${slice.state}`}>{stateLabel(slice.state, vocabulary, locale)}</span>
      <strong className="numeric" dir="ltr">{countedText(slice, counted, locale)}</strong>
      <small>{daysText(slice, locale)}</small>
      {stillToCome ? (
        <small>
          {pageText(
            locale,
            `Still to come: ${stillToCome} ${spotWord(stillToCome, locale)}`,
            `עוד לפנינו: ⁦${stillToCome}⁩ ${spotWord(stillToCome, locale)}`,
          )}
        </small>
      ) : null}
    </span>
  );
}

// The as-of instant as the ledger recorded it, with the sentence the ledger
// recorded beside it. That sentence crosses the wire in one language only,
// because nothing translated it, so it is marked as the source's own wording
// rather than presented as this product's Hebrew.
function AsOf({ asOf, locale }) {
  const instant = String((asOf && asOf.instant) || '').trim();
  const basis = String((asOf && asOf.basis) || '').trim();
  if (!instant) {
    return null;
  }
  return (
    <p className="clients-basis-note">
      <span>{pageText(locale, 'Counted as of', 'נספר נכון ל־')}</span>
      {pageText(locale, ' ', '')}
      <span className="numeric" dir="ltr">{instant}</span>
      {basis ? (
        <>
          {'. '}
          <span lang="en" dir="ltr">{basis}</span>
        </>
      ) : '.'}
    </p>
  );
}

// Everything a counted figure on this surface was counted on. It renders in both
// states on purpose: when no day has a source it names the missing feed and the
// path that supplies it, and when days do have a source it names the instant,
// the file, the days nobody has a source for and the rule that removed spots.
export function DeliveryBasis({ delivery, locale }) {
  // No ledger reached this surface at all, which is a different state from a
  // ledger that reports nothing. It is stated rather than left silent, because
  // the alternative is the word unknown standing on a row with no reason under
  // it, which is the defect this component was built to end.
  if (!delivery) {
    return (
      <p className="clients-basis-note">
        {pageText(
          locale,
          'The delivery ledger was not read on this screen, so what aired is unknown here rather than counted.',
          'ספר האספקה לא נקרא במסך הזה, ולכן מה ששודר אינו ידוע כאן ואינו נספר.',
        )}
      </p>
    );
  }
  if (!delivery.available) {
    return (
      <>
        <p className="clients-basis-note">{localized(delivery, 'reason', locale)}</p>
        <p className="clients-basis-path">{localized(delivery, 'path_forward', locale)}</p>
      </>
    );
  }

  const slice = deliverySlice(delivery);
  const files = sourceFilesOf(slice);
  const rules = droppedRulesOf(slice);
  const dropped = slice.aired.droppedByRule + slice.scheduled.droppedByRule;
  const floor = localized(delivery.unknown, 'reason', locale) || localized(delivery, 'floor_note', locale);

  return (
    <>
      <AsOf asOf={delivery.as_of} locale={locale} />
      {floor ? <p className="clients-basis-note">{floor}</p> : null}
      {slice.unknownDays > 0 ? (
        <p className="clients-basis-note">
          <span>
            {pageText(
              locale,
              `${slice.unknownDays} flight days carry no per-spot source and are not counted as zero:`,
              `⁦${slice.unknownDays}⁩ ימי טיסה ללא מקור ברמת התשדיר, ואינם נספרים כאפס:`,
            )}
          </span>
          {' '}
          <span className="numeric" dir="ltr">{slice.unknownDates.join(', ')}</span>
        </p>
      ) : null}
      {files.length ? (
        <p className="clients-basis-note">
          <span>{pageText(locale, 'The file these counts were read out of:', 'הקובץ שממנו נקראו הספירות האלה:')}</span>
          {' '}
          <span dir="ltr">{files.join(', ')}</span>
        </p>
      ) : null}
      {dropped > 0 ? (
        <p className="clients-basis-note">
          <span>
            {pageText(
              locale,
              `Removed by a rule on the counted days: ${dropped} ${spotWord(dropped, locale)}. The count above is short by that many.`,
              `הוסרו על ידי כלל בימים שנספרו: ⁦${dropped}⁩ ${spotWord(dropped, locale)}. הספירה שלמעלה חסרה במספר הזה.`,
            )}
          </span>
          {rules.length ? <span dir="ltr">{` ${rules.join(', ')}`}</span> : null}
        </p>
      ) : null}
      <p className="clients-basis-note">
        {pageText(
          locale,
          'Times are as the source file records them. No time zone is declared on this ledger.',
          'השעות הן כפי שקובץ המקור רושם אותן. לא מוצהר אזור זמן בספר הזה.',
        )}
      </p>
    </>
  );
}

// One goal and how far the counted figures have got against it. The percent is
// never printed alone: it carries the endpoint's own state word, and a floor is
// read out as a floor in the figure itself rather than only in a note. When
// there is no percent the reason stands in its place, because a goal nobody can
// measure against is a state and not a zero.
function Goal({ label, reading, reason, basis, locale }) {
  return (
    <div className={reading.hasPercent ? 'clients-progress' : 'clients-progress empty'}>
      <dt>{label}</dt>
      <dd>
        {reading.hasPercent ? (
          <>
            <strong className="numeric" dir="ltr">
              {reading.isFloor
                ? pageText(locale, `at least ${decimals(reading.percent, 2, locale)}%`, `לפחות ⁦${decimals(reading.percent, 2, locale)}%⁩`)
                : `${decimals(reading.percent, 2, locale)}%`}
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
