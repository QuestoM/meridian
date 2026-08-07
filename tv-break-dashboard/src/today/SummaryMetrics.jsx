import React from 'react';
import { CircleDollarSign, Clock3, ShieldCheck, Users } from 'lucide-react';
import {
  finiteNumber,
  formatCurrency,
  formatMinutes,
  formatNumber,
  formatPercent,
  pageText,
  summaryBasisLabel,
} from '../shell/format';
import { formatSpan } from '../shell/dates';
import { riskLabel } from '../shell/labels';
import { Metric } from '../shell/primitives';
import { PlanEventBadges } from '../rules/CalendarEventsModel';
import ChannelRefusal from './ChannelRefusal';
import { overviewScope, unattributed } from './today-scope';

export function SummaryMetrics({ overview, copy, locale, planEvents = null, onOpenSettings = null }) {
  // A malformed-but-online response falls back to an empty summary so the
  // metrics show honest empty states, never the offline demo numbers.
  const summary = overview.summary || {};
  // Whose four figures these are. The overview body summed the whole market
  // whenever it could not scope, and it says so in the same breath, so the
  // strip refuses rather than printing four rivals' aggregate under this
  // operator's name. The basis note below goes with it: it is the sentence that
  // would name the channel, and it silently drops that clause instead.
  const scope = overviewScope(overview);
  if (unattributed(scope)) {
    return (
      <>
        <ChannelRefusal
          locale={locale}
          lead={pageText(
            locale,
            'These four figures cannot be reported as yours yet.',
            'אי אפשר עדיין לדווח על ארבעת המספרים האלה כשלכם.',
          )}
          onOpenSettings={onOpenSettings}
        />
        <PlanEventBadges events={planEvents} locale={locale} />
      </>
    );
  }
  // The headline speaks in the operator's working horizon: the planning-week
  // slice the API computes (summary.week). Whole-plan totals stay available as
  // the top-level summary keys and serve as the fallback for an older backend.
  const week = summary.week && typeof summary.week === 'object' ? summary.week : null;
  const revenueValue = week ? week.projected_revenue : summary.projected_revenue;
  const retentionValue = week ? week.average_retention : summary.average_retention;
  const adSecondsValue = week ? week.total_ad_seconds : summary.total_ad_seconds;
  const riskScore = finiteNumber(week ? week.risk_score : summary.risk_score);
  const basisLabel = summaryBasisLabel(summary, locale);
  const revenueHint = pageText(
    locale,
    'Projected ad revenue of your channel for the planning week, summed from the optimizer’s saved plan. Not a forecast for tomorrow.',
    'צפי הכנסות הפרסום של הערוץ שלכם לשבוע התכנון, מסוכם מהתוכנית השמורה של האופטימייזר. לא תחזית ליום מחר.',
  );
  // The exact week window and its per-day pace live inside the tile itself:
  // a bare total reads as an opaque blob, and the operator thinks in days.
  const weekDates = week ? finiteNumber(week.n_dates) : null;
  const weekRevenue = week ? finiteNumber(week.projected_revenue) : null;
  const weekRange = week && week.date_from && week.date_to
    ? formatSpan(week.date_from, week.date_to, locale)
    : null;
  const revenueSub = week && weekRange && weekDates > 0 && weekRevenue !== null
    ? pageText(
      locale,
      `${weekRange} · daily average ${formatCurrency(weekRevenue / weekDates, locale)}`,
      `${weekRange} · ממוצע יומי ${formatCurrency(weekRevenue / weekDates, locale)}`,
    )
    : null;
  // Retention here is a PER-BREAK, momentary measure, and the copy must say so:
  // read as a weekly churn figure ("we lose 5% of viewers every week?") it is
  // both alarming and wrong. The tooltip and the sub line carry the honest
  // reading: audience dip during a break, mostly returning after it.
  const retentionHint = pageText(
    locale,
    'How much of the programme audience stays through an ad break, TVR-weighted across the planning week. A momentary per-break measure, not a cumulative weekly loss of viewers: people who step away during a break mostly come back to the programme, but the break itself airs to a smaller audience, and that is the revenue it costs. This is the audience side of the retention cost in the money story.',
    'כמה מקהל התוכנית נשאר מול המסך במהלך ברייק פרסומות, ממוצע משוקלל TVR על ברייקי שבוע התכנון. זהו מדד רגעי לכל ברייק, לא איבוד צופים מצטבר משבוע לשבוע: מי שעוזב בזמן ברייק לרוב חוזר לתוכנית מיד אחריו, אבל הברייק עצמו משודר לקהל קטן יותר, וזו ההכנסה שהוא מפסיד. זהו הצד הקהלי של עלות השימור בסיפור הכסף.',
  );
  const retentionPct = finiteNumber(retentionValue);
  const retentionSub = retentionPct !== null
    ? pageText(
      locale,
      `about ${formatNumber(Math.round((100 - retentionPct) * 10) / 10, locale)}% step away during an average break, mostly returning after it`,
      `כ-${formatNumber(Math.round((100 - retentionPct) * 10) / 10, locale)}% עוזבים זמנית בברייק ממוצע, ורובם חוזרים מיד אחריו`,
    )
    : null;
  const minutesHint = pageText(
    locale,
    'Total ad seconds on the planning week of the saved plan, shown as minutes.',
    'סך שניות הפרסום בשבוע התכנון של התוכנית השמורה, מוצג בדקות.',
  );
  const riskHint = pageText(
    locale,
    'How far the planning week’s average retention sits below your retention floor (0 = at or above the floor). Not a general business-risk score.',
    'כמה רחוק ממוצע השימור של שבוע התכנון מתחת לרף השימור שלכם (0 = ברף או מעליו). לא ציון סיכון עסקי כללי.',
  );
  return (
    <>
      <section className="metric-strip" aria-label="Optimization summary">
        <Metric label={copy.metrics[0]} value={formatCurrency(revenueValue, locale)} sub={revenueSub} icon={CircleDollarSign} positive title={revenueHint} />
        <Metric label={copy.metrics[1]} value={formatPercent(retentionValue, locale)} sub={retentionSub} icon={Users} title={retentionHint} />
        <Metric label={copy.metrics[2]} value={formatMinutes(adSecondsValue, locale)} icon={Clock3} positive title={minutesHint} />
        <Metric label={copy.metrics[3]} value={riskScore === null ? '-' : copy.risk[riskLabel(riskScore)]} delta={riskScore === null ? '-' : `${riskScore}/100`} icon={ShieldCheck} tone="risk" title={riskHint} />
      </section>
      {basisLabel && (
        <p className="data-basis-note">
          {week && weekRange
            ? pageText(
              locale,
              `The headline figures are the projection for the planning week ${weekRange} (${formatNumber(weekDates, locale)} days), taken from the saved plan for ${basisLabel}.`,
              `המספרים שבכותרת הם הצפי לשבוע התכנון ${weekRange} (${formatNumber(weekDates, locale)} ימים), מתוך התוכנית השמורה עבור ${basisLabel}.`,
            )
            : pageText(
              locale,
              `These headline figures are totals from the saved plan for ${basisLabel}. They are not a day-ahead forecast.`,
              `המספרים שבכותרת הם סכומים מהתוכנית השמורה עבור ${basisLabel}. זו אינה תחזית ליום מחר.`,
            )}
        </p>
      )}
      <PlanEventBadges events={planEvents} locale={locale} />
    </>
  );
}

export default SummaryMetrics;
