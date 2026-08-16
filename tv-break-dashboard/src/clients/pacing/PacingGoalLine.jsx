import React from 'react';
import { Button } from '../../studio/actions';
import { Figure } from '../../shell/bidi';
import { ILS, bare, isolate, localized, opensDays, pair, percent, pick, vocabularyLabel } from './pacing-helpers';

// The campaign's other goal, stated rather than hidden.
//
// A campaign here is booked against two goals and the payload has carried both
// on every row since the first round: a rating goal in points and a money goal
// in shekels, each with its own counted figure, its own even-share reference and
// its own verdict. The row printed one of them and the drill printed a bare pair
// of the other with no verdict, no reference and no ratio.
//
// Measured on the shipped board, which is why this is not a nicety. 48 of the 56
// rows carry both goals, and on 10 of those 48 the two verdicts disagree.
// Every one of the 10 is a row the board is asking a decision about: it reads at
// risk on rating and on pace on money, CMP_D040 at 0.88 against 0.9989. That is
// the difference between a campaign that is spending to plan and under-
// delivering audience, which is a conversation with the buyer, and a campaign
// that is behind on both, which is a booking problem. An account manager pacing
// a flight answers for the money as well as for the rating, which is what
// Google Ads, the reference for this piece, is a spend-against-budget board
// first.
//
// One component, rendered on the row and again under the day drill, so this
// product cannot come to hold two statements of one goal.

function goalWord(unit, locale) {
  if (unit === ILS) return pick(locale, 'The money goal', 'היעד הכספי');
  return pick(locale, 'The rating goal', 'יעד הרייטינג');
}

export default function PacingGoalLine({
  line, vocabulary, locale, className = '', days = 0, expanded = false, onOpen = null,
}) {
  if (!line || line.goal === null || line.goal === undefined) return null;
  const figures = pair(line.counted.through_counted_day, line.goal, line.unit, locale);
  if (figures === null) return null;
  const verdict = line.pace ? line.pace.verdict : '';
  const ratio = line.pace ? percent(line.pace.ratio, locale) : null;
  const reference = line.reference
    ? bare(line.reference.expected_through_counted_day, line.unit, locale)
    : null;
  // A second line whose pace could not be stated says which one is missing. The
  // headline line prints its own reason above the track and this one has nowhere
  // else to print it, so a reader never meets the word unknown on its own.
  const why = ratio === null ? localized(line.pace, 'reason', locale) : '';
  const verdicts = (vocabulary || {}).pace_verdicts;
  return (
    <div className={`pacing-goal-line ${className}`.trim()}>
      <span className="pacing-goal-word">{goalWord(line.unit, locale)}</span>
      {/* This goal's figure is the way into the days it was summed from, exactly
          as the headline figure above it is. The two amounts on a row were one
          control and one piece of inert text, so Stripe's transferable mechanic
          had been applied to whichever goal the row happened to lead with. Both
          are summed from the same broadcast days, so both open the same drill
          and both carry the same accessible name, written in one place.

          Under the drill it is text again. The line is rendered a second time
          inside the days it would open, and a disclosure control sitting inside
          the thing it discloses is a control that can only close it. */}
      {onOpen && days ? (
        <Button type="button" className="pacing-goal-figures pacing-figure-open"
                aria-expanded={expanded} aria-label={opensDays(figures, days, locale)}
                onClick={onOpen}>
          {figures}
        </Button>
      ) : <span className="pacing-goal-figures">{figures}</span>}
      {/* The ratio and the verdict are two facts and take a rule between them.
          In English they were one span and read as "100% On pace", a capitalised
          label concatenated onto a figure, which reads as a title rather than as
          a reading. The Hebrew form never had the problem and is unchanged. */}
      {ratio ? <Figure className={`pacing-goal-ratio ${verdict}`}>{ratio}</Figure> : null}
      <span className={`pacing-goal-verdict ${verdict}`}>
        {vocabularyLabel(verdicts, verdict, locale)}
      </span>
      {ratio && reference !== null ? (
        <span className="pacing-goal-against">
          {pick(
            locale,
            `against a reference of ${reference} by that day`,
            `מול ייחוס של ${isolate(reference)} עד אותו יום`,
          )}
        </span>
      ) : null}
      {why ? <span className="pacing-goal-why">{why}</span> : null}
    </div>
  );
}
