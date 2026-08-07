// What the weekday chips under a discount percent mean, in one place.
//
// Two screens carry this pair, the campaign amend form and the onboarding flow,
// and each had its own copy of the sentence. Both copies were wrong in the same
// way and had to be corrected twice, which is the argument for one copy.
//
// The endpoint's rule is check_weekday_scope, and it has two halves: a percent
// that is neither zero nor blank, and no weekday to spend it on. It returns
// without refusing anything when the percent is missing. A sentence that reads
// only the chips promised a refusal that never came, so an operator amending the
// notes of a campaign that has no discount was told the save would be refused
// while it saved. The percent is read here for that reason.

import { pageText } from '../shell/format';

const NO_WEEKDAY_AGENCY = [
  'No weekday is selected. Submitting like this is refused: an agency condition with no weekday scope covers every day.',
  'לא נבחר יום בשבוע. שליחה כך תסורב: תנאי סוכנות ללא היקף ימים חל על כל יום.',
];
const NO_WEEKDAY_TERM = [
  'No weekday is selected. Submitting like this is refused: the discount percent would have no day it covers.',
  'לא נבחר יום בשבוע. שליחה כך תסורב: אחוז ההנחה יהיה ללא יום שהוא חל עליו.',
];
const NO_DISCOUNT_TO_SCOPE = [
  'No weekday is selected. Nothing is refused, because there is no discount percent to give a day to.',
  'לא נבחר יום בשבוע. דבר אינו נדחה, כיוון שאין אחוז הנחה שצריך לתת לו יום.',
];

// Whether the endpoint would really refuse this pair. The same coercion the
// endpoint makes: a blank, a null and an unparseable value are all no discount.
export function wouldRefuse(selected, percent) {
  const amount = Number(percent);
  return !(selected || []).length && Number.isFinite(amount) && amount !== 0;
}

export function weekdayCoverage(selected, options, locale, { asAgencyRule = false, percent = null } = {}) {
  const chosen = selected || [];
  if (!chosen.length) {
    if (!wouldRefuse(chosen, percent)) {
      return pageText(locale, ...NO_DISCOUNT_TO_SCOPE);
    }
    return pageText(locale, ...(asAgencyRule ? NO_WEEKDAY_AGENCY : NO_WEEKDAY_TERM));
  }
  const names = (options || [])
    .filter((day) => chosen.includes(day.key))
    .map((day) => (locale === 'he' ? day.he : day.en));
  return pageText(locale, `Covers ${names.join(', ')}.`, `חל על ${names.join(', ')}.`);
}
