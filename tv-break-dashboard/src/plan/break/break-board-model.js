// What the break board's footer is allowed to say, held apart from the board.
//
// The rule is one line long and it is the whole point of this module: the figure
// under a column is the sum of the rows in that column, and never anything else.
// It lives here rather than inline in the board because the defect it replaces
// was an inline expression, and an expression inside JSX cannot be driven by a
// test. Every function below is plain data in and data out, with no import, so
// the module a browser runs is the module a test executes.
//
// Measured on רשת 13 / 2024-11-01 with one break marked gold, which marks the
// three breaks of its programme: the gold filter leaves three rows printing
// 10,712, 10,163 and 9,614, and the line under them used to print 1,028,206, the
// whole day, 33.7 times the column it claimed to total.

// The rows the filter leaves. The board renders these and the footer totals
// these, from the one call, so the two can never describe different sets.
export function visibleRows(breaks, goldOnly) {
  const list = breaks || [];
  return goldOnly ? list.filter((row) => row.is_gold) : list;
}

// The sum of what is displayed, taken on the full credits rather than on the
// rounded ones. The route serves each break to the agora and the board prints
// whole shekels, so the two differ: measured on the three gold rows above,
// 10,711.71 + 10,162.61 + 9,613.52 is 30,487.84, which prints as 30,488, while
// the printed column adds by eye to 30,489. Summing the full values is what
// keeps this footer equal to the day's own revenue when nothing is filtered, and
// equal to what the day board and every other surface print for the same day.
// The one shekel the eye is missing is named on the surface, in roundingSentence
// below, rather than left for a reader to find.
export function sumRevenue(rows) {
  return (rows || []).reduce((sum, row) => sum + Number(row.projected_revenue || 0), 0);
}

// The share of the day this column carries, or null when there is no day to take
// a share of. A zero denominator is unknown and never zero percent.
export function shareOfDay(shown, dayRevenue) {
  const day = Number(dayRevenue);
  if (!Number.isFinite(day) || day <= 0) return null;
  return (Number(shown) / day) * 100;
}

// What the column under the table is, in words, and it changes when the filter
// changes it. The unfiltered sentence is the claim the footer can be checked
// against; the filtered one names the subset, its share of the day and where the
// day's own figure is, so no sentence on this surface ever describes a column
// that is not on screen. The share arrives already formatted, and null when it
// cannot be computed, because this module holds no formatter.
export function basisSentence({ goldOnly, shownCount, total, portion, locale }) {
  const he = locale === 'he';
  const credit = he
    ? 'כל נתון הוא הזיכוי שהמנוע ייחס לברייק באותו ערוץ ואותו יום.'
    : 'Every figure is the optimizer credit to that break on this channel-day.';
  if (!goldOnly) {
    const whole = he
      ? 'הסינון כבוי, ולכן הטור הזה הוא כל הברייקים ביום, והסכום שמתחתיו הוא הכנסת היום.'
      : 'The filter is off, so this column is every break in the day and the sum under it is the day.';
    return `${credit} ${whole}`;
  }
  if (he) {
    const carried = portion ? `, והם נושאים ${portion} מהכנסת היום` : '';
    return `${credit} סינון ברייקי הזהב פעיל, ולכן הטור הזה מחזיק ${shownCount} ברייקים מתוך ${total}${carried}. הכנסת היום המלאה היא השורה השנייה בתחתית.`;
  }
  const carried = portion ? `, carrying ${portion} of the day` : '';
  return `${credit} The gold filter is on, so this column holds ${shownCount} breaks of ${total}${carried}. The day's own revenue is the second line at the foot.`;
}

// Where the rounding actually goes, said out loud on the surface that does it.
//
// Measured across all thirty planned days: the day's own revenue and its breaks
// added up agree to the shekel on twenty nine of them and are one shekel apart
// on 2024-11-22, while hand-adding the printed column can be out by up to five.
export function roundingSentence(locale) {
  if (locale === 'he') {
    return 'כל שורה מעוגלת לשקל השלם, ולכן חיבור ידני של הטור עשוי להיבדל מהסכום בכמה שקלים. הסכום עצמו מחושב על הערכים המלאים.';
  }
  return 'Each row is rounded to the whole shekel, so adding the column by hand can differ from the sum by a few shekels. The sum itself is computed on the full values.';
}
