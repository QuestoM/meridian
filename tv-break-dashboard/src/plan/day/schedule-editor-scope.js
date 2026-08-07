// What this timeline draws against what the saved weekly plan actually holds.
//
// Measured by a critic on רשת 13 / 2024-11-01, over four rounds, and each round
// closed one clause and left the next one standing.
//
// Round one: three break counts stood on one screen with no sentence connecting
// them and no route from the drawn 8 to the rest of the day or from that one day
// to the other 29. Round two: the sentence arrived, and its numerator and its
// denominator came from two different plans. The drawn chips are built by the
// frozen plan_read.build_break_operations from output/weekly_break_schedule.csv,
// the SAVED weekly plan, 80 breaks that day; the denominator was read off
// score.current.breaks, the LIVE re-plan, 76 breaks that day, and 0 of the 8
// drawn clock times sat among those 76. Round three read both counts off
// score.basis.committed, so the containment finally held.
//
// Round four measured what was left, and it is what this module now closes.
//
// The saved plan places 13 breaks in the 12 programmes this timeline draws, and
// the timeline draws 8 of them: 02:00 one placed and none drawn, 02:17 one and
// one, 02:41 two and one, 03:10 four and three, 04:08 one and one, 04:29 one and
// none, 05:01 two and two, 05:40 one and none. The old sentence counted those 5
// undrawn breaks as part of the 72 "rest" and pointed at a link for them, and
// the link opens the day board, which re-plans the day live and holds a
// different 76. So the sentence sent a scheduler after 5 breaks that sit inside
// programmes already on the screen, down a route that does not carry them.
//
// The 13 is read from the saved plan's own num_breaks, joined to the drawn
// programmes on exactly the key build_break_operations joins on, (channel, date,
// HH:MM), through the anchors this component already fetches. Nothing new is
// scored and no route is added. The drawn list itself is untouched: the
// twelve-programme cap is the frozen plan_read, which this piece does not own,
// so the fix is the true sentence, not a bigger timeline.
//
// Every clause names the plan it counts, through plan-basis.js, because the
// money tiles below this sentence count the live plan and the two were being
// read as one number.

import { useEffect, useState } from 'react';
import { fetchDays } from './day-board-actions.js';
import { LIVE_PLAN, SAVED_PLAN, planBasisLabel, planBasisLead } from './plan-basis.js';

// What the saved plan places in the programmes this timeline draws, by lane and
// in total, plus how many of those programmes the plan carries a row for at all.
//
// A drawn programme with no saved row is real: the EPG feed and the plan are two
// files. It is counted in what is shown and it is absent from what the plan
// holds, so it is reported separately rather than quietly inflating either side.
export function plannedInShown(programs, resolve) {
  const rows = Array.isArray(programs) ? programs : [];
  const byLane = {};
  let total = 0;
  let matched = 0;
  rows.forEach((program) => {
    if (!program) return;
    const hit = typeof resolve === 'function' ? resolve(program.channel, program.date, program.start_time) : null;
    const planned = hit ? Number(hit.plannedBreaks) : NaN;
    if (!Number.isFinite(planned)) return;
    matched += 1;
    total += planned;
    const lane = program.lane || '';
    byLane[lane] = (byLane[lane] || 0) + planned;
  });
  return { total, matched, byLane };
}

export function buildCoverage({ breaksShown, programs, score, daysInPlan, resolve, anchorsLoaded }) {
  const rows = Array.isArray(programs) ? programs : [];
  const committed = score ? score.basis.committed : null;
  const breaksInDay = committed ? Number(committed.breaks) : null;
  const programmesInDay = committed ? Number(committed.segments) : null;
  const live = score && score.current ? Number(score.current.breaks) : null;
  const day = score ? score.basis.day : '';
  const placed = anchorsLoaded ? plannedInShown(rows, resolve) : null;
  // The dates the drawn chips actually sit on. The sentence takes its day from
  // the score, and a capped board is free in principle to draw two dates, so
  // this is measured rather than assumed.
  const daysDrawn = Array.from(new Set(rows.map((program) => (program && program.date) || '').filter(Boolean))).sort();
  return {
    breaksShown: Number(breaksShown) || 0,
    programsShown: rows.length,
    plannedInShown: placed ? placed.total : null,
    programmesMatched: placed ? placed.matched : null,
    plannedByLane: placed ? placed.byLane : {},
    breaksInDay: Number.isFinite(breaksInDay) ? breaksInDay : null,
    programmesInDay: Number.isFinite(programmesInDay) ? programmesInDay : null,
    liveBreaksInDay: Number.isFinite(live) ? live : null,
    daysInPlan: Number.isFinite(daysInPlan) ? daysInPlan : null,
    day,
    daysDrawn,
    // True only once the score has answered and it named no committed plan for
    // this channel-day, which is the one state this sentence must not paper
    // over with a figure borrowed from the live re-plan.
    committedUnavailable: Boolean(score) && !committed,
  };
}

// The one sentence, in both directions, with the plan named in every clause that
// carries a number. Silent about any figure that has not answered yet rather
// than naming a total it does not have.
export function coverageSentence(coverage, locale) {
  const label = (en, he) => (locale === 'he' ? he : en);
  const saved = planBasisLabel(SAVED_PLAN, locale);
  const {
    breaksShown, programsShown, plannedInShown: placed, programmesMatched,
    breaksInDay, programmesInDay, daysInPlan, day, daysDrawn, committedUnavailable,
  } = coverage;
  if (committedUnavailable) {
    const lead = planBasisLead(SAVED_PLAN, locale);
    return label(
      `${lead} carries no committed figures for this day${day ? ` (${day})` : ''}, so what this timeline draws cannot be checked against it.`,
      `${lead} אינה מחזיקה נתונים מחויבים ליום הזה${day ? ` (${day})` : ''}, ולכן אי אפשר לבדוק מולה מה הרצועה הזו מציגה.`,
    );
  }
  if (breaksInDay === null || programmesInDay === null) {
    return label('Reading how much of the day this timeline draws.', 'קורא כמה מהיום הרצועה הזו מציגה.');
  }
  const shownProgrammes = programmesMatched === null ? programsShown : programmesMatched;
  // The covers clause. Until the plan's own per-programme counts have answered
  // there is no 13 to name, so the clause states what it does have, against the
  // day total, and names the plan for both.
  const breaksPart = placed === null
    ? label(
        `This timeline draws ${breaksShown} of the ${breaksInDay} breaks ${saved} holds for this day`,
        `הרצועה הזו מציגה ${breaksShown} מתוך ${breaksInDay} הברייקים ש${saved} מחזיקה ליום הזה`,
      )
    : label(
        `This timeline draws ${breaksShown} of the ${placed} breaks ${saved} places in the ${shownProgrammes} programmes it shows, out of the ${breaksInDay} that plan holds for this day`,
        `הרצועה הזו מציגה ${breaksShown} מתוך ${placed} הברייקים ש${saved} קובעת ל-${shownProgrammes} התוכניות שהיא מציגה, מתוך ${breaksInDay} שאותה תוכנית מחזיקה ליום הזה`,
      );
  const programsPart = label(
    `across ${shownProgrammes} of its ${programmesInDay} programmes`,
    `על פני ${shownProgrammes} מתוך ${programmesInDay} התוכניות שלה`,
  );
  const dayPart = daysInPlan
    ? label(
        `on 1 of its ${daysInPlan} days${day ? ` (${day})` : ''}`,
        `ביום 1 מתוך ${daysInPlan} ימיה${day ? ` (${day})` : ''}`,
      )
    : label(`on this one day${day ? ` (${day})` : ''}`, `ביום הזה בלבד${day ? ` (${day})` : ''}`);
  return `${breaksPart}, ${programsPart}, ${dayPart}.${unmatchedClause(coverage, locale)}${straddleClause(daysDrawn, day, locale)}`;
}

// The drawn programmes the saved plan carries no row for. Counted in what is
// shown, absent from what the plan holds, and named rather than netted away.
function unmatchedClause(coverage, locale) {
  const { programsShown, programmesMatched } = coverage;
  if (programmesMatched === null) return '';
  const missing = programsShown - programmesMatched;
  if (missing <= 0) return '';
  const saved = planBasisLabel(SAVED_PLAN, locale);
  return locale === 'he'
    ? ` עוד ${missing} תוכניות מוצגות כאן ש${saved} אינה מחזיקה להן שורה.`
    : ` ${missing} more programmes are drawn here that ${saved} carries no row for.`;
}

// The sentence names one day. If the drawn chips ever straddle two dates it says
// so instead of letting one date stand for both.
function straddleClause(daysDrawn, day, locale) {
  const dates = Array.isArray(daysDrawn) ? daysDrawn.filter((value) => value && value !== day) : [];
  if (!dates.length) return '';
  return locale === 'he'
    ? ` הברייקים המוצגים משתרעים גם על ${dates.join(', ')}.`
    : ` The breaks drawn also fall on ${dates.join(', ')}.`;
}

// Where the link goes, and what is actually there.
//
// The board behind it re-plans this day live against current settings, models
// and constraints, so it holds its own set of breaks and not the rest of these.
// Measured on רשת 13 / 2024-11-01: 76 chips there, 0 of them at any of the 8
// clock times drawn here. This states both counts and claims no containment
// between them, which is the only honest thing to say about two plans.
export function routeSentence(coverage, locale) {
  const live = coverage.liveBreaksInDay;
  const saved = coverage.breaksInDay;
  if (!Number.isFinite(live) || !Number.isFinite(saved)) {
    return locale === 'he'
      ? `לוח היום המלא מתכנן את היום הזה מחדש בזמן אמת מול ההגדרות הנוכחיות, ולכן הוא מחזיק ברייקים משלו ולא את אלה.`
      : `The full day board re-plans this day live against current settings, so it holds breaks of its own rather than the rest of these.`;
  }
  const liveNamed = planBasisLabel(LIVE_PLAN, locale);
  return locale === 'he'
    ? `לוח היום המלא מתכנן את היום הזה מחדש בזמן אמת מול ההגדרות הנוכחיות: ${liveNamed} מחזיקה שם ${live} ברייקים משלה, מול ${saved} שנספרו כאן.`
    : `The full day board re-plans this day live against current settings: ${liveNamed} holds ${live} breaks of its own there, against the ${saved} counted here.`;
}

// One fetch of the plan's own day count, held for the life of the editor. The
// timeline itself never asks for more than the twelve programmes it draws, so
// without this the sentence above could name what is drawn and nothing to
// compare it against.
export function useEditorCoverage({ breaksShown, programs, score, resolve, anchorsLoaded }) {
  const [daysInPlan, setDaysInPlan] = useState(null);
  useEffect(() => {
    let alive = true;
    fetchDays()
      .then((payload) => { if (alive) setDaysInPlan(payload && payload.count); })
      .catch(() => { if (alive) setDaysInPlan(null); });
    return () => { alive = false; };
  }, []);
  return buildCoverage({ breaksShown, programs, score, daysInPlan, resolve, anchorsLoaded });
}
