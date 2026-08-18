import React from 'react';
import { Figure, Code, isolate } from '../shell/bidi';
import { pageText } from '../shell/format';
import { localized } from './clients-money-helpers';
import { formatDayList, formatStamp } from '../shell/dates';
import { deliverySlice, droppedRulesOf, sourceFilesOf, spotWord } from './delivery-helpers';

// The BASIS half of this destination's delivery display: every sentence that
// says what a counted figure was counted on. The figures themselves are in
// DeliveryState.jsx, and the two are one contract — a figure is rendered by
// DeliveryCell or DeliveryProgress and its basis by DeliveryBasis on the same
// surface, so no count can reach a reader without the instant it was taken at,
// the file it came from and the days nobody has a source for.
//
// Split out of DeliveryState.jsx when that file passed this destination's
// 450-line cap. The seam moved; the law did not. The test that holds it shut
// (tests/test_p4_delivery_on_screen.py) names both files as the ledger's owners
// and still refuses any other surface the ledger's raw field names.

// The as-of instant as the ledger recorded it, with the sentence the ledger
// recorded beside it. Prefer a server-supplied Hebrew basis. The seeded basis
// predates that field, so its one known sentence is translated here; any future
// untranslated server wording remains explicitly marked as English.
function AsOf({ asOf, locale }) {
  const instant = String((asOf && asOf.instant) || '').trim();
  const basis = String((asOf && asOf.basis) || '').trim();
  const suppliedHebrew = String((asOf && asOf.basis_he) || '').trim();
  const visibleBasis = locale === 'he'
    ? suppliedHebrew || ({
      'The start of the last programme booked on the newest sourced broadcast day, so the demo shows what has aired and what is still to come on that day.': 'נקודת הספירה היא תחילת התוכנית האחרונה ששובצה ביום השידור העדכני ביותר שיש עבורו מקור. לכן ההדגמה מפרידה באותו יום בין מה ששודר לבין מה שעדיין מתוכנן.',
    }[basis] || basis)
    : basis;
  const basisLanguage = locale === 'he' && visibleBasis !== basis ? 'he' : 'en';
  if (!instant) {
    return null;
  }
  return (
    <p className="clients-basis-note">
      <span>{pageText(locale, 'Counted as of', 'נספר נכון ל־')}</span>
      {pageText(locale, ' ', '')}
      <Figure className="numeric">{formatStamp(instant) || instant}</Figure>
      {visibleBasis ? (
        <>
          {'. '}
          <span lang={basisLanguage}>{visibleBasis}</span>
        </>
      ) : '.'}
    </p>
  );
}

// What a rule that removed spots actually capped, in a reader's words.
//
// Three states, and the third is the point. A rule the server composed reads as
// its cap; a rule the rule file does not hold says exactly that and carries the
// path that would fix it, both in the server's wording; and a block that never
// arrived says the rule is unnamed here rather than inventing a cap for it or
// falling back to the engine key this whole helper exists to keep off the
// screen.
function ruleSentence(block, locale) {
  const sentence = localized(block, 'rule', locale);
  if (sentence) {
    return sentence;
  }
  return pageText(
    locale,
    'A booking rule removed them. Which rule is not named on this screen.',
    'כלל הזמנה הסיר אותם. איזה כלל — אינו נקוב במסך הזה.',
  );
}

function rulePath(block, locale) {
  return localized(block, 'path_forward', locale);
}

// What that rule cost, in the sentence that names the rule.
//
// It cost MONEY and not the count, which is the opposite of what this used to
// say. The sentence read "the count above is short by that many", and the count
// above is not short: the ledger's spot column is the number of rows the traffic
// file carries for that campaign and day, dropped ones included, while its
// spend column is the engine's price for the spots that survived the rule.
// Measured on the shipped store, and it is an identity rather than an
// impression: for all 41 clients with a source, ledger spots equals the money
// layer's priced spots plus its dropped spots exactly, and the two layers'
// dropped counts are equal to the row. On פריסבי that is 9 airings, 6 priced,
// 3 removed by DEFAULT_ONE_PER_BREAK — and the drawer told a reader the 9 was
// missing 3. The money layer's own note beside the figures has always said this
// correctly; now both halves of the screen say the same thing.
function droppedText(dropped, locale) {
  const one = dropped === 1;
  return pageText(
    locale,
    one
      ? `It took ${dropped} ${spotWord(dropped, locale)} out of the pricing, so its money is not in the figures. The spot count above includes it.`
      : `It took ${dropped} ${spotWord(dropped, locale)} out of the pricing, so their money is not in the figures. The spot count above includes them.`,
    one
      ? `הוא הוציא ${isolate(dropped)} ${spotWord(dropped, locale)} מהתמחור, ולכן הכסף שלו אינו בסכומים. ספירת התשדירים שלמעלה כוללת אותו.`
      : `הוא הוציא ${isolate(dropped)} ${spotWord(dropped, locale)} מהתמחור, ולכן הכסף שלהם אינו בסכומים. ספירת התשדירים שלמעלה כוללת אותם.`,
  );
}

// The three sentences that belong to the LEDGER READ and not to any one row:
// the instant the aired/scheduled split was taken at, that every figure counted
// this way is a floor, and that the times are the source file's own with no
// zone declared. One ledger read produces one of each.
//
// They live apart from DeliveryBasis because a surface that lists many rows was
// printing all three under every one of them. Measured on the clients drawer:
// two campaigns, six paragraphs, of which four were the same two paragraphs
// twice — and a client with a dozen campaigns repeats the same 240 characters a
// dozen times, which is how a screen teaches a reader to stop reading it.
//
// A surface that shows ONE row keeps them inline (DeliveryBasis renders them by
// default), because there the repetition does not exist and splitting the basis
// off the figure would break the law this file is built on: no count reaches a
// reader without the basis it was counted on beside it.
export function DeliveryLedgerNote({ ledger, locale, ratingBasis = false }) {
  if (!ledger || !ledger.available) {
    return null;
  }
  const floor = localized(ledger, 'floor_note', locale);
  // Only when a rating figure is actually on the surface. The ledger's rating
  // column is the PLANNED break rating out of the traffic log and not a panel
  // report of delivered points, and the endpoint publishes that sentence with
  // the figure. A counted rating point may not reach a reader without it, and a
  // screen showing no rating point should not carry a caveat about one.
  const rating = ratingBasis ? localized(ledger, 'rating_basis', locale) : '';
  return (
    <div className="clients-ledger-note">
      <AsOf asOf={ledger.as_of} locale={locale} />
      {floor ? <p className="clients-basis-note">{floor}</p> : null}
      {rating ? <p className="clients-basis-note">{rating}</p> : null}
      <p className="clients-basis-note">
        {pageText(
          locale,
          'Times are as the source file records them. No time zone is declared on this ledger.',
          'השעות הן כפי שקובץ המקור רושם אותן. לא מוצהר אזור זמן בספר הזה.',
        )}
      </p>
    </div>
  );
}

// Everything a counted figure on this surface was counted on. It renders in both
// states on purpose: when no day has a source it names the missing feed and the
// path that supplies it, and when days do have a source it names the instant,
// the file, the days nobody has a source for and the rule that removed spots.
//
// ``ledgerNote`` false leaves out the three sentences DeliveryLedgerNote states
// once for the whole read. A caller that turns it off owes the reader that
// component somewhere above these rows; a caller that leaves it alone gets the
// complete basis inline, which is what a single-row surface needs.
export function DeliveryBasis({ delivery, locale, ledgerNote = true, ratingBasis = false }) {
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
  const rules = droppedRulesOf(slice, delivery.booking_rules);
  const dropped = slice.aired.droppedByRule + slice.scheduled.droppedByRule;
  const floor = localized(delivery.unknown, 'reason', locale) || localized(delivery, 'floor_note', locale);

  return (
    <>
      {ledgerNote ? <AsOf asOf={delivery.as_of} locale={locale} /> : null}
      {ledgerNote && floor ? <p className="clients-basis-note">{floor}</p> : null}
      {/* Same law as DeliveryLedgerNote's: a counted rating point may not reach
          a reader without the ledger's sentence about what its rating column
          is. A surface that answers a GRP goal in GRP passes this true. */}
      {ledgerNote && ratingBasis && localized(delivery, 'rating_basis', locale) ? (
        <p className="clients-basis-note">{localized(delivery, 'rating_basis', locale)}</p>
      ) : null}
      {slice.unknownDays > 0 ? (
        <p className="clients-basis-note">
          <span>
            {pageText(
              locale,
              `${slice.unknownDays} flight days carry no per-spot source and are not counted as zero:`,
              `${isolate(slice.unknownDays)} ימי טיסה ללא מקור ברמת התשדיר, ואינם נספרים כאפס:`,
            )}
          </span>
          {' '}
          {/* Not wrapped in Figure, and not clickable, and both were measured.
              Figure forces left-to-right, which is right for one quantity and
              wrong for a list, because it would put the earliest run on the far
              side of a Hebrew line; formatDayList isolates each run on its own
              and leaves the ORDER to the line. And a click here does nothing:
              this is a span inside a paragraph with no handler on it or on any
              of its three call sites (CampaignBoard, ClientRecord,
              CampaignFlights), so merging a run into a range takes nothing away
              from a reader. A day rendered as a control it is not would. */}
          {formatDayList(slice.unknownDates, locale)}
        </p>
      ) : null}
      {files.length ? (
        <p className="clients-basis-note">
          <span>{pageText(locale, 'The file these counts were read out of:', 'הקובץ שממנו נקראו הספירות האלה:')}</span>
          {' '}
          <Code>{files.join(', ')}</Code>
        </p>
      ) : null}
      {/* One rule is one sentence: what it caps and what it cost, together. The
          two were separate paragraphs, which on a client with four campaigns
          meant eight lines saying what four could. They are only split when
          MORE THAN ONE rule dropped spots here, because then the count is a
          total across rules and attaching it to any single rule's sentence
          would say that rule dropped all of them. */}
      {dropped > 0 && rules.length === 1 ? (
        <p className="clients-basis-note">
          <span>{ruleSentence(rules[0].block, locale)}</span>
          {' '}
          <span>{droppedText(dropped, locale)}</span>
          {rulePath(rules[0].block, locale) ? (
            <>
              {' '}
              <span className="clients-basis-path">{rulePath(rules[0].block, locale)}</span>
            </>
          ) : null}
        </p>
      ) : null}
      {dropped > 0 && rules.length !== 1 ? (
        <>
          <p className="clients-basis-note">
            {pageText(
              locale,
              `Taken out of the pricing by a rule on the counted days: ${dropped} ${spotWord(dropped, locale)}. Their money is not in the figures, and the spot count above includes them.`,
              `הוצאו מהתמחור על ידי כלל בימים שנספרו: ${isolate(dropped)} ${spotWord(dropped, locale)}. הכסף שלהם אינו בסכומים, וספירת התשדירים שלמעלה כוללת אותם.`,
            )}
          </p>
          {rules.map(({ id, block }) => (
            <p className="clients-basis-note" key={id}>
              <span>{ruleSentence(block, locale)}</span>
              {rulePath(block, locale) ? (
                <>
                  {' '}
                  <span className="clients-basis-path">{rulePath(block, locale)}</span>
                </>
              ) : null}
            </p>
          ))}
        </>
      ) : null}
      {ledgerNote ? (
        <p className="clients-basis-note">
          {pageText(
            locale,
            'Times are as the source file records them. No time zone is declared on this ledger.',
            'השעות הן כפי שקובץ המקור רושם אותן. לא מוצהר אזור זמן בספר הזה.',
          )}
        </p>
      ) : null}
    </>
  );
}
