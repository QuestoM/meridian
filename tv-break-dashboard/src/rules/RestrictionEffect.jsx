import React from 'react';
import { AlertCircle, Loader2 } from 'lucide-react';
import { pageText } from '../shell/format';
import { Figure as BidiFigure, Name } from '../shell/bidi';
import { clock, collateralSentence, dayLabel, isolate, lengthLabel, money, pairLabel, valuePair } from './rules-lib';

// The effect panel. It exists because a restriction is a decision about somebody
// else's revenue and the person making it has never been shown the number. Two
// figures are reported and they are never blended: what the removed breaks were
// carrying, and what the plan would be worth if it were run with this rule in
// place. Each carries its scope and its basis on the figure, not in a tooltip.
//
// The two do not start from the same number, because one counts from the plan as
// saved and the other from a run today. That is stated on each figure and the
// gap between them is printed, so a reader is never left holding two starting
// points that disagree with nothing to explain them.
//
// The change list is in two parts, because a rule can move an airing its own
// sentence never named. The airings the sentence asked for come first; anything
// the compiled predicate reached beyond them is under its own heading with its
// own money, so the surplus is a stated fact and not a share of one total.

function ChangeRow({ change, locale }) {
  return (
    <li>
      <span className="rules-change-day">
        {dayLabel(change.day, locale)}
        <small><BidiFigure>{clock(change.start_seconds)}</BidiFigure></small>
      </span>
      <Name className="rules-change-title">{change.title}</Name>
      <BidiFigure className="rules-change-length">{lengthLabel(change.duration_seconds, locale)}</BidiFigure>
      <BidiFigure
        className="rules-change-breaks"
        aria-label={pairLabel(locale, change.before_breaks, change.after_breaks)}
      >
        {valuePair(change.before_breaks, change.after_breaks)}
      </BidiFigure>
    </li>
  );
}

function ChangeList({ changes, locale, limit = 8 }) {
  if (!changes.length) return null;
  return (
    <ul className="rules-effect-changes">
      {changes.slice(0, limit).map((change) => (
        <ChangeRow key={change.segment_id} change={change} locale={locale} />
      ))}
      {changes.length > limit && (
        <li className="rules-effect-more">
          {pageText(
            locale,
            `${changes.length - limit} more airings change the same way.`,
            `עוד ${changes.length - limit} שידורים משתנים באותו אופן.`,
          )}
        </li>
      )}
    </ul>
  );
}

function Figure({ locale, label, before, after, delta, scope, basis, startingPoint }) {
  const negative = Number(delta) < 0;
  const from = money(before, locale);
  const to = money(after, locale);
  return (
    <div className="card rules-figure">
      <span className="rules-figure-label">{label}</span>
      <strong className={`rules-figure-delta${negative ? ' negative' : ' positive'}`}>
        <BidiFigure>{isolate(money(delta, locale))}</BidiFigure>
      </strong>
      <BidiFigure className="rules-figure-pair" aria-label={pairLabel(locale, from, to)}>
        {valuePair(from, to)}
      </BidiFigure>
      <span className="rules-figure-start">{startingPoint}</span>
      <span className="rules-figure-scope">{scope}</span>
      <span className="rules-figure-basis">{basis}</span>
    </div>
  );
}

export default function RestrictionEffect({ locale, preview, previewing, error, sayable }) {
  if (!sayable) {
    return (
      <div className="rules-effect rules-effect-idle">
        {pageText(
          locale,
          'Pick a programme and the cost of this restriction appears here.',
          'בחרו תוכנית והעלות של ההגבלה תופיע כאן.',
        )}
      </div>
    );
  }
  if (error) {
    return (
      <div className="rules-effect rules-effect-error" role="status">
        <AlertCircle size={15} aria-hidden="true" />
        <span>{error}</span>
      </div>
    );
  }
  if (previewing && !preview) {
    return (
      <div className="rules-effect rules-effect-idle" role="status">
        <Loader2 size={15} className="rules-spin" aria-hidden="true" />
        <span>{pageText(locale, 'Pricing this restriction', 'מתמחר את ההגבלה')}</span>
      </div>
    );
  }
  if (!preview) return null;

  const changes = preview.changes || [];
  // A change the sentence named, against a change a compiled predicate reached
  // anyway. The server decides which is which; the panel never guesses.
  const asked = changes.filter((change) => change.asked_for !== false);
  const surplus = changes.filter((change) => change.asked_for === false);
  const collateral = preview.collateral || {};
  const collateralMoney = collateral.revenue?.available
    ? money(collateral.revenue.revenue_delta, locale)
    : '';
  const scored = preview.scored || {};
  const exact = preview.exact || {};
  const starts = preview.starting_points || {};
  const skipped = preview.engine_skipped || [];
  const refused = exact.refusals || [];
  const unknown = Number(preview.airings_without_a_plan || 0);
  // Every unavailable basis carries its reason in both languages. A money panel
  // on an operator surface never prints the other language's sentence, and never
  // an internal key.
  const reasonOf = (side) => (locale === 'he' ? side.reason_he : side.reason_en) || '';
  const scopeText = pageText(
    locale,
    `${preview.channel}, ${scored.days || exact.days || 0} broadcast days`,
    `${preview.channel}, ${scored.days || exact.days || 0} ימי שידור`,
  );

  return (
    <div className={`rules-effect${previewing ? ' rules-effect-stale' : ''}`}>
      <p className="rules-effect-sentence">
        {locale === 'he' ? preview.sentence_he : preview.sentence_en}
      </p>

      <p className="rules-effect-count">
        {pageText(
          locale,
          `This rule matches ${preview.matched_airings} airings and binds ${preview.bound_airings} of them across ${preview.bound_days} broadcast days. The break count changes on ${changes.length}.`,
          `הכלל הזה תואם ${preview.matched_airings} שידורים ומחייב ${preview.bound_airings} מהם ב-${preview.bound_days} ימי שידור. מספר הברייקים משתנה ב-${changes.length}.`,
        )}
      </p>

      {unknown > 0 && (
        <p className="rules-effect-unknown" role="status">
          {pageText(
            locale,
            `${unknown} of the airings this rule binds carry no break count in the plan of record, so their cost is unknown rather than zero. Run the weekly plan to fill them.`,
            `ל-${unknown} מהשידורים שהכלל מחייב אין מספר ברייקים בתוכנית הרשומה, ולכן העלות שלהם אינה ידועה ולא אפס. הריצו את התוכנית השבועית כדי למלא אותם.`,
          )}
        </p>
      )}

      {skipped.length > 0 && (
        <p className="rules-effect-unknown" role="status">
          {pageText(
            locale,
            `The plan engine refused this rule on ${skipped.length} airings: ${reasonOf(skipped[0])}.`,
            `מנוע התוכנית דחה את הכלל הזה ב-${skipped.length} שידורים: ${reasonOf(skipped[0])}.`,
          )}
        </p>
      )}

      <ChangeList changes={asked} locale={locale} />

      {collateral.applies && Number(collateral.bound || 0) > 0 && (
        <div className="rules-effect-collateral">
          <p className="rules-effect-collateral-note" role="status">
            {collateralSentence(locale, collateral, collateralMoney)}
          </p>
          <ChangeList changes={surplus} locale={locale} limit={4} />
        </div>
      )}

      <div className="rules-figures">
        {scored.available ? (
          <Figure
            locale={locale}
            label={pageText(locale, 'The breaks this removes were carrying', 'הברייקים שיוסרו נשאו')}
            before={scored.revenue_before}
            after={scored.revenue_after}
            delta={scored.revenue_delta}
            scope={scopeText}
            startingPoint={locale === 'he' ? scored.starting_point_he : scored.starting_point_en}
            basis={pageText(
              locale,
              'Scored at the counts this rule sets, before the plan is run again.',
              'מחושב לפי מספרי הברייקים שהכלל קובע, לפני הרצה מחדש של התוכנית.',
            )}
          />
        ) : (
          <div className="card rules-figure rules-figure-empty">
            <span className="rules-figure-label">{pageText(locale, 'The breaks this removes were carrying', 'הברייקים שיוסרו נשאו')}</span>
            <span className="rules-figure-reason">{reasonOf(scored)}</span>
          </div>
        )}

        {exact.available ? (
          <Figure
            locale={locale}
            label={pageText(locale, 'The plan after it is run with this rule', 'התוכנית אחרי הרצה עם הכלל הזה')}
            before={exact.revenue_before}
            after={exact.revenue_after}
            delta={exact.revenue_delta}
            scope={pageText(
              locale,
              `${preview.channel}, ${exact.days} broadcast days`,
              `${preview.channel}, ${exact.days} ימי שידור`,
            )}
            startingPoint={locale === 'he' ? exact.starting_point_he : exact.starting_point_en}
            basis={pageText(
              locale,
              'The optimizer run twice on those days, so a break can move elsewhere.',
              'האופטימייזר רץ פעמיים על אותם ימים, כך שברייק יכול לעבור למקום אחר.',
            )}
          />
        ) : (
          <div className="card rules-figure rules-figure-empty">
            <span className="rules-figure-label">{pageText(locale, 'The plan after it is run with this rule', 'התוכנית אחרי הרצה עם הכלל הזה')}</span>
            <span className="rules-figure-reason">{reasonOf(exact)}</span>
          </div>
        )}
      </div>

      {refused.length > 0 && (
        <p className="rules-effect-unknown" role="status">
          {pageText(
            locale,
            `The plan will not carry this rule on ${refused.length} of the airings it binds: ${reasonOf(refused[0])}. The scored figure prices the arrangement the rule asks for, not the plan that would result from it.`,
            `התוכנית לא תישא את הכלל הזה ב-${refused.length} מהשידורים שהוא מחייב: ${reasonOf(refused[0])}. המספר המחושב מתמחר את הסידור שהכלל מבקש, לא את התוכנית שתתקבל ממנו.`,
          )}
        </p>
      )}

      {preview.already_in_force?.all && (
        <p className="rules-inline-note" role="status">
          {pageText(
            locale,
            'This rule is already in force, so running the plan with it changes nothing further. Saving it again would write a second copy of the same rule.',
            'הכלל הזה כבר בתוקף, ולכן הרצת התוכנית איתו לא תשנה דבר נוסף. שמירה חוזרת תיצור עותק שני של אותו כלל.',
          )}
        </p>
      )}

      {starts.comparable && !starts.same_start && (
        <p className="rules-effect-gap">
          {locale === 'he' ? starts.note_he : starts.note_en}
          {' '}
          {pageText(
            locale,
            `The gap between the two starting points is ${money(starts.gap, locale)}.`,
            `הפער בין שתי נקודות הפתיחה הוא ${money(starts.gap, locale)}.`,
          )}
        </p>
      )}

      {exact.available && Number.isFinite(Number(exact.retention_after)) && (
        <p className="rules-effect-retention">
          {pageText(
            locale,
            `Viewer retention moves from ${(Number(exact.retention_before) * 100).toFixed(2)}% to ${(Number(exact.retention_after) * 100).toFixed(2)}% on those days.`,
            `שימור הצופים עובר מ-${(Number(exact.retention_before) * 100).toFixed(2)}% ל-${(Number(exact.retention_after) * 100).toFixed(2)}% באותם ימים.`,
          )}
        </p>
      )}
    </div>
  );
}
