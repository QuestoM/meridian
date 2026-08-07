// What this timeline draws against what the day and the plan actually hold.
//
// Measured by a critic on רשת 13 / 2024-11-01: the timeline drew 8 of the day's
// 76 breaks across 12 of the day's 82 programmes, on 1 of the plan's 30 days, and
// priced the whole day directly beneath it without ever saying so. Three break
// counts stood on one screen with no sentence connecting them and no route from
// the drawn 8 to the other 68 or from that one day to the other 29.
//
// A second critic measured the first fix and found the sentence still false: the
// drawn 8 chips are built by the frozen plan_read.build_break_operations from
// output/weekly_break_schedule.csv, the saved weekly plan (80 breaks that day),
// while the denominator this module read, score.current.breaks, is the live
// re-plan (76 breaks that day) built from current settings and models. Zero of
// the 8 drawn clock times sit among the live 76, so "8 of 76" stated a
// containment that did not hold. The numerator and the denominator have to come
// from the same plan. score.basis.committed carries exactly that plan (the same
// output/weekly_break_schedule.csv row group build_break_operations reads), so
// this module reads breaksInDay and programmesInDay from there and no longer
// from the live re-plan. When this channel-day carries no committed row at all
// -- which cannot happen while chips are drawn, since both read the same file,
// but is checked rather than assumed -- the sentence says so instead of
// borrowing a number from a different plan.
//
// The fix does not touch the drawn list itself: the twelve-programme cap is
// section 8.2's frozen plan_read.build_break_operations, and this piece does not
// own it. What this piece owns is the sentence, built from figures the page
// already fetched: the drawn lists (breaksShown / programsShown), the committed
// weekly plan the money panel already scored (score.basis.committed) and the
// plan's own day count, read once here through the day board's own fetchDays.

import { useEffect, useState } from 'react';
import { fetchDays } from './day-board-actions.js';

export function buildCoverage({ breaksShown, programsShown, score, daysInPlan }) {
  const committed = score ? score.basis.committed : null;
  const breaksInDay = committed ? Number(committed.breaks) : null;
  const programmesInDay = committed ? Number(committed.segments) : null;
  const day = score ? score.basis.day : '';
  return {
    breaksShown: Number(breaksShown) || 0,
    programsShown: Number(programsShown) || 0,
    breaksInDay: Number.isFinite(breaksInDay) ? breaksInDay : null,
    programmesInDay: Number.isFinite(programmesInDay) ? programmesInDay : null,
    daysInPlan: Number.isFinite(daysInPlan) ? daysInPlan : null,
    day,
    // True only once the score has answered and it named no committed plan for
    // this channel-day, which is the one state this sentence must not paper
    // over with a figure borrowed from the live re-plan.
    committedUnavailable: Boolean(score) && !committed,
  };
}

// The one sentence, in both directions. Silent about the plan's day count when
// that figure has not answered yet, rather than naming a total it does not have.
export function coverageSentence(coverage, locale) {
  const label = (en, he) => (locale === 'he' ? he : en);
  const {
    breaksShown, programsShown, breaksInDay, programmesInDay, daysInPlan, day, committedUnavailable,
  } = coverage;
  if (committedUnavailable) {
    return label(
      `The saved weekly plan carries no committed figures for this day${day ? ` (${day})` : ''}, so what this timeline draws cannot be checked against it.`,
      `לתוכנית השבועית השמורה אין נתונים מחויבים ליום הזה${day ? ` (${day})` : ''}, ולכן אי אפשר לבדוק מול מה הרצועה הזו מציגה.`,
    );
  }
  if (breaksInDay === null || programmesInDay === null) {
    return label('Reading how much of the day this timeline draws.', 'קורא כמה מהיום הרצועה הזו מציגה.');
  }
  const breaksPart = label(
    `This timeline draws ${breaksShown} of the day's ${breaksInDay} breaks`,
    `הרצועה הזו מציגה ${breaksShown} מתוך ${breaksInDay} הברייקים של היום`,
  );
  const programsPart = label(
    `across ${programsShown} of ${programmesInDay} programmes`,
    `על פני ${programsShown} מתוך ${programmesInDay} תוכניות`,
  );
  const dayPart = daysInPlan
    ? label(
        `on 1 of the plan's ${daysInPlan} days${day ? ` (${day})` : ''}`,
        `ביום 1 מתוך ${daysInPlan} בתוכנית${day ? ` (${day})` : ''}`,
      )
    : label(`on this one day${day ? ` (${day})` : ''}`, `ביום הזה בלבד${day ? ` (${day})` : ''}`);
  return `${breaksPart}, ${programsPart}, ${dayPart}.`;
}

// One fetch of the plan's own day count, held for the life of the editor. The
// timeline itself never asks for more than the twelve programmes it draws, so
// without this the sentence above could name what is drawn and nothing to
// compare it against.
export function useEditorCoverage({ breaksShown, programsShown, score }) {
  const [daysInPlan, setDaysInPlan] = useState(null);
  useEffect(() => {
    let alive = true;
    fetchDays()
      .then((payload) => { if (alive) setDaysInPlan(payload && payload.count); })
      .catch(() => { if (alive) setDaysInPlan(null); });
    return () => { alive = false; };
  }, []);
  return buildCoverage({ breaksShown, programsShown, score, daysInPlan });
}
