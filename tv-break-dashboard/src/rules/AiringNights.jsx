import React from 'react';
import { pageText } from '../shell/format';
import { dayLabel, nightAriaLabel, nightDetail, nightsHeadSentence } from './rules-lib';

// Which night this restriction is about. Every night the programme runs, none
// hidden, because the scope a person can name is a date and the count above the
// chips is the count of choices below them.
//
// The defect this replaces was a cap. The picker rendered the first twelve
// airings and nothing else: no search, no date field, no disclosure and no count
// of what was missing. Measured on the reference plan for
// משחקי השף עונה 7 ש.ח, the head line read forty-three and the twelve chips
// under it covered six of the nineteen nights, so thirteen nights were
// unreachable, the two Sunday airings sat at positions twenty-four and
// twenty-five, and the last night of the run could not be named at all. Across
// the operator's forty titles, twenty-six ran more than twelve airings and 1,896
// airings sat behind the cap.
//
// Deduplicating by night is what makes rendering all of them small. Two airings
// on one night compile to the same predicate, so they were always one choice
// shown twice, and a broadcast month bounds the list at thirty whatever the
// programme: the busiest title on the channel is 1,551 airings on 5 nights.

export default function AiringNights({ locale, airings, nights, day, onPick }) {
  const list = nights || [];
  return (
    <div className="rules-airings">
      <span className="rules-airings-head">
        {nightsHeadSentence(airings, list.length, locale)}
      </span>
      <div className="rules-airing-chips" role="group">
        <button
          type="button"
          className={`rules-airing-chip${day ? '' : ' active'}`}
          aria-pressed={!day}
          onClick={() => onPick('')}
        >
          {pageText(locale, 'Every airing', 'כל השידורים')}
        </button>
        {list.map((night) => (
          <button
            key={night.day}
            type="button"
            className={`rules-airing-chip${day === night.day ? ' active' : ''}`}
            aria-pressed={day === night.day}
            aria-label={nightAriaLabel(night, locale)}
            onClick={() => onPick(night.day)}
          >
            <span>{dayLabel(night.day, locale)}</span>
            <small dir="ltr">{nightDetail(night, locale)}</small>
          </button>
        ))}
      </div>
    </div>
  );
}
