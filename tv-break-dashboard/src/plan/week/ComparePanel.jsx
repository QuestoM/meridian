import React from 'react';
import { Button } from '@mui/material';
import { ArrowLeft, ArrowRight, Check, GitCompare, RefreshCcw, X } from 'lucide-react';
import { finiteNumber, formatCurrency, formatNumber, formatPercent, pageText } from '../../shell/format';
import { Figure, Name } from '../../shell/bidi';
import ScenarioLegForm from './ScenarioLegForm';
import CompareWeekTable from './CompareWeekTable';
import { leverLabel } from './plan-week-model';
import ScenarioAdopt from './ScenarioAdopt';

// Step three, and the reason this destination exists.
//
// JS-2 defines the planner's comparison on revenue net of retention cost, over
// next week. The panel used to fail all three: it printed "Net after retention
// cost: Not exposed", it printed a delta of zero on every operational figure,
// and it compared one representative broadcast day while every other figure on
// this destination was the week.
//
// All three are closed and all three are measurements. The net is computed by
// the engine's own per-break retention-cost model, the same basis the committed
// plan's yield-per-second money uses, so expected revenue minus cost equals net
// by construction. Every lever is on both legs, because measured on
// רשת 13 / 2024-11-11 revenue weight 60 and 85 return the identical plan and
// only the floor, the hourly cap, the caution and the engine focus move it. And
// both legs now run the plan's own week, the same seven dates the goal strip
// reports, arriving one broadcast day at a time because fourteen real
// optimizations are 11 to 13 seconds and a spinner would say nothing about
// where they are.

// The blended score lives between 0 and 1, and the shell formatter keeps one
// decimal, which prints 0.5405 and 0.4611 as the same "0.5". Two scenarios that
// differ only in the score would then read as identical, so it is printed at
// the grain the comparison is made at.
function blendedScore(value, locale) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return number.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', {
    minimumFractionDigits: 3,
    maximumFractionDigits: 3,
  });
}

function MoneyRow({ label, value, locale, tone }) {
  return (
    <div className={`plan-money-row${tone ? ` ${tone}` : ''}`}>
      <span>{label}</span>
      <strong className="numeric"><Figure>{formatCurrency(value, locale)}</Figure></strong>
    </div>
  );
}

export function ScenarioCard({ leg, title, summary, accent, locale, words, scopeText, windowText, onAdopt }) {
  if (!summary) return null;
  const money = summary.money_available === true;
  return (
    <div className={`plan-scenario-card${accent ? ` ${accent}` : ''}`}>
      <div className="plan-scenario-head">
        <strong>{title}</strong>
        <Figure className="numeric">
          {leverLabel('revenue_weight', locale)} {finiteNumber(summary.levers?.revenue_weight) ?? '-'}
        </Figure>
      </div>
      {money ? (
        <div className="plan-money-block">
          {windowText ? <p className="plan-money-window">{windowText}</p> : null}
          <MoneyRow label={words.expectedRevenue} value={summary.gross} locale={locale} />
          <MoneyRow label={words.retentionCost} value={summary.retention_cost} locale={locale} tone="cost" />
          <MoneyRow
            label={pageText(locale, 'Net after retention cost', 'נטו אחרי עלות שימור')}
            value={summary.revenue_net}
            locale={locale}
            tone="net"
          />
          {scopeText ? <p className="plan-scope-line">{scopeText}</p> : null}
        </div>
      ) : (
        <p className="plan-note plan-note-amber">
          {pageText(
            locale,
            'Net after retention cost cannot be computed for this run, so no money figure is shown.',
            'לא ניתן לחשב נטו אחרי עלות שימור עבור ההרצה הזאת, ולכן לא מוצג ערך כספי.',
          )}
          {summary.money_reason ? <small className="plan-note-detail"><Name>{summary.money_reason}</Name></small> : null}
        </p>
      )}
      <dl className="plan-scenario-stats">
        <div>
          <dt>{pageText(locale, 'Average retention', 'שימור ממוצע')}</dt>
          <dd className="numeric"><Figure>{formatPercent(summary.average_retention, locale)}</Figure></dd>
        </div>
        <div>
          <dt>{words.breaks}</dt>
          <dd className="numeric"><Figure>{formatNumber(summary.total_breaks, locale)}</Figure></dd>
        </div>
        <div>
          <dt>{pageText(locale, 'Ad seconds', 'שניות פרסום')}</dt>
          <dd className="numeric"><Figure>{formatNumber(summary.total_ad_seconds, locale)}</Figure></dd>
        </div>
        <div>
          <dt>
            {summary.objective_basis === 'mean_of_days'
              ? pageText(locale, 'Blended score, mean of the days', 'ציון משוקלל, ממוצע הימים')
              : pageText(locale, 'Blended score', 'ציון משוקלל')}
          </dt>
          <dd className="numeric"><Figure>{blendedScore(summary.objective, locale)}</Figure></dd>
        </div>
      </dl>
      <span className={`plan-compliance${summary.compliant ? ' ok' : ' warn'}`}>
        {summary.compliant
          ? pageText(locale, 'Inside every guardrail', 'בתוך כל המגבלות')
          : pageText(locale, 'Breaches a guardrail', 'חורג ממגבלה')}
      </span>
      {onAdopt ? <ScenarioAdopt leg={leg} summary={summary} locale={locale} onAdopt={onAdopt} /> : null}
    </div>
  );
}

function DeltaRow({ label, value, locale, formatter, suffix, emphasis }) {
  const number = finiteNumber(value);
  if (number === null) {
    return (
      <div className="plan-delta-row">
        <span>{label}</span>
        <strong>{pageText(locale, 'not available', 'לא זמין')}</strong>
      </div>
    );
  }
  const sign = number > 0 ? '+' : '';
  const tone = number > 0 ? 'up' : number < 0 ? 'down' : '';
  return (
    <div className={`plan-delta-row${emphasis ? ' is-headline' : ''}`}>
      <span>{label}</span>
      <strong className={`numeric ${tone}`}><Figure>{sign}{formatter(number, locale)}{suffix || ''}</Figure></strong>
    </div>
  );
}

// What the run cost, in the plain terms it was measured in. Fourteen real
// optimizations are slower than JS-2's five-second bar and saying so beats a
// silent wait, so the panel prints the runs, the clock and how many of them were
// reused from this session rather than computed again.
function runCostLine(scope, locale) {
  const runs = scope?.runs;
  const elapsed = finiteNumber(scope?.elapsed_ms);
  if (!runs || elapsed === null) return null;
  const seconds = (elapsed / 1000).toFixed(1);
  const en = `${runs.total} optimizer runs, ${seconds} s. ${runs.computed} computed now, ${runs.reused} reused from this session.`;
  const he = `${runs.total} ריצות אופטימייזר, ${seconds} שניות. ${runs.computed} חושבו עכשיו, ${runs.reused} נלקחו מהרצה קודמת במושב הזה.`;
  return pageText(locale, en, he);
}

// What the destination is doing with the machine before it is asked.
//
// The comparison is fourteen real optimizations and 12.6 s cold, so it is
// started while the planner is still setting the scenarios up. That spends a
// machine's time on their behalf, and spending it silently would not be honest,
// so the panel says which of the two states it is in and never claims a figure
// from work that has not finished.
function prepareLine(phase, locale) {
  if (phase === 'preparing') {
    return pageText(
      locale,
      'Both scenarios are being computed in the background while you set them up.',
      'שני התרחישים מחושבים ברקע בזמן שאתם מכווננים אותם.',
    );
  }
  if (phase === 'ready') {
    return pageText(
      locale,
      'Both scenarios are already computed for these settings, so the comparison returns without another wait.',
      'שני התרחישים כבר חושבו בהגדרות האלה, ולכן ההשוואה תחזור בלי המתנה נוספת.',
    );
  }
  return null;
}

export function ComparePanel({
  locale, words, legA, legB, state, payload, error, runWindow, liveDays, prepared,
  onLegChange, onCompare, onCancel, onAdopt, onOpenDay,
}) {
  const he = locale === 'he';
  const Flow = he ? ArrowLeft : ArrowRight;
  const running = state === 'running';
  const ready = state === 'ready' && payload;
  const scope = ready ? payload.scope : null;
  const week = scope?.mode === 'week';
  const dates = running ? (runWindow?.dates || []) : (scope?.dates || []);
  const days = running ? liveDays : (payload?.by_day || []);
  const channel = running ? runWindow?.channel : scope?.channel;
  // The window each money figure was summed over, printed above the figures
  // rather than beside one of them, because all three share it.
  const ranDays = finiteNumber(payload?.a?.days);
  const shortOfWindow = week && ranDays !== null && ranDays !== finiteNumber(scope?.n_dates);
  const windowText = !ready
    ? null
    : shortOfWindow
      ? pageText(
        locale,
        `${formatNumber(ranDays, locale)} of the plan’s own ${formatNumber(scope.n_dates, locale)} broadcast days ran`,
        `רצו ${formatNumber(ranDays, locale)} מתוך ${formatNumber(scope.n_dates, locale)} ימי השידור של שבוע התוכנית`,
      )
      : week
        ? pageText(
          locale,
          `The plan’s own week, ${formatNumber(scope.n_dates, locale)} broadcast days`,
          `שבוע התוכנית עצמו, ${formatNumber(scope.n_dates, locale)} ימי שידור`,
        )
        : pageText(locale, `One broadcast day, ${scope?.day || ''}`, `יום שידור אחד, ${scope?.day || ''}`);
  // The channel name carries its own direction inside a sentence in the other
  // language, so the scope line is assembled from parts rather than one string.
  const scopeText = !ready || !scope?.channel
    ? null
    : (
      <>
        {pageText(locale, 'Your channel ', 'הערוץ שלכם ')}
        <bdi>{scope.channel}</bdi>
        {week ? pageText(locale, ', ', ', ') : pageText(locale, ', broadcast day ', ', יום שידור ')}
        <bdi>{week ? `${scope.date_from} ${pageText(locale, 'to', 'עד')} ${scope.date_to}` : scope.day}</bdi>
        {pageText(
          locale,
          `, ${formatNumber(scope.segments, locale)} programme segments`,
          `, ${formatNumber(scope.segments, locale)} מקטעי תוכנית`,
        )}
      </>
    );
  // The day that separates the two scenarios most. Read off the rows that came
  // back, never a guess about which day matters.
  const biggest = (days || []).reduce((best, row) => {
    const value = Math.abs(Number(row?.delta_revenue_net) || 0);
    return value > 0 && value > Math.abs(Number(best?.delta_revenue_net) || 0) ? row : best;
  }, null);
  const runCost = ready ? runCostLine(scope, locale) : null;
  const preparing = running ? null : prepareLine(prepared, locale);

  return (
    <section className="plan-section" aria-labelledby="plan-compare-title">
      <div className="plan-section-head">
        <div>
          <h2 id="plan-compare-title">{pageText(locale, 'Compare two ways to run it', 'השוואה בין שתי דרכים להריץ')}</h2>
          <p>
            {ready && !week
              ? pageText(
                locale,
                'Two real optimizer runs on one representative broadcast day, side by side, judged on revenue net of retention cost. Neither run touches the saved plan.',
                'שתי ריצות אמיתיות של האופטימייזר על יום שידור מייצג אחד, זו לצד זו, נמדדות לפי הכנסה בניכוי עלות שימור. אף אחת מהן אינה נוגעת בתוכנית השמורה.',
              )
              : pageText(
                locale,
                'Two real optimizer runs over the plan’s own week, side by side, judged on revenue net of retention cost. Neither run touches the saved plan.',
                'שתי ריצות אמיתיות של האופטימייזר על שבוע התוכנית עצמו, זו לצד זו, נמדדות לפי הכנסה בניכוי עלות שימור. אף אחת מהן אינה נוגעת בתוכנית השמורה.',
              )}
          </p>
        </div>
        <div className="plan-compare-actions">
          <Button className="run-button" type="button" variant="contained" disabled={running} onClick={onCompare}>
            {running ? <RefreshCcw size={15} className="upload-spinner" /> : <GitCompare size={15} />}
            {running ? pageText(locale, 'Comparing', 'משווה') : pageText(locale, 'Compare', 'השוואה')}
          </Button>
          {running && onCancel ? (
            <Button type="button" variant="text" onClick={onCancel}>
              <X size={15} />
              {pageText(locale, 'Stop', 'עצירה')}
            </Button>
          ) : null}
        </div>
      </div>

      {preparing ? (
        <p className={`plan-prepare-note is-${prepared}`} role="status">
          {prepared === 'preparing' ? <RefreshCcw size={13} className="upload-spinner" /> : <Check size={13} />}
          <span>{preparing}</span>
        </p>
      ) : null}

      <div className="plan-leg-grid">
        <ScenarioLegForm
          leg="a"
          title={pageText(locale, 'Scenario A', 'תרחיש A')}
          values={legA}
          locale={locale}
          onChange={(field, value) => onLegChange('a', field, value)}
        />
        <ScenarioLegForm
          leg="b"
          title={pageText(locale, 'Scenario B', 'תרחיש B')}
          values={legB}
          locale={locale}
          onChange={(field, value) => onLegChange('b', field, value)}
        />
      </div>

      {error && (
        <p className="plan-note plan-note-red" role="alert">
          {pageText(locale, `The comparison could not run: ${error}`, `ההשוואה לא הצליחה לרוץ: ${error}`)}
        </p>
      )}
      {state === 'unavailable' && (
        <p className="plan-note plan-note-amber" role="status">
          {payload?.reason || pageText(locale, 'The comparison is unavailable.', 'ההשוואה אינה זמינה.')}
        </p>
      )}

      {running && (
        <>
          {runWindow ? (
            <p className="plan-note plan-note-quiet" role="status">
              {pageText(locale, 'Running both scenarios over ', 'מריץ את שני התרחישים על ')}
              {channel ? <bdi>{channel}</bdi> : null}
              {pageText(
                locale,
                `, ${formatNumber(dates.length, locale)} broadcast days. Each day appears the moment both scenarios have decided it.`,
                `, ${formatNumber(dates.length, locale)} ימי שידור. כל יום מופיע ברגע ששני התרחישים הכריעו אותו.`,
              )}
            </p>
          ) : (
            <p className="plan-note plan-note-quiet" role="status">
              {pageText(
                locale,
                'Running both scenarios over the plan’s own week. This browser could not take the day-by-day stream, so the whole week arrives at once.',
                'מריץ את שני התרחישים על שבוע התוכנית עצמו. הדפדפן הזה לא הצליח לקבל את הזרימה יום-אחר-יום, ולכן כל השבוע מגיע בבת אחת.',
              )}
            </p>
          )}
          <CompareWeekTable
            locale={locale}
            dates={dates}
            days={liveDays || []}
            running
            elapsedMs={(liveDays || []).length ? liveDays[liveDays.length - 1].elapsed_ms : null}
            biggestDate={biggest?.date || null}
            onOpenDay={onOpenDay}
          />
        </>
      )}

      {ready && (
        <>
          {!week && (
            <p className="plan-note plan-note-amber" role="status">
              {pageText(locale, 'One representative broadcast day, not the week.', 'יום שידור מייצג אחד, ולא השבוע.')}
              {scope?.day_reason ? <small className="plan-note-detail"><Name>{scope.day_reason}</Name></small> : null}
              <small className="plan-note-detail">
                {pageText(
                  locale,
                  'The weekly comparison runs on the plan’s own week. Run the plan on step 2 so the week has dates, then compare again.',
                  'ההשוואה השבועית רצה על שבוע התוכנית עצמו. הריצו את התוכנית בשלב 2 כדי שיהיו לשבוע תאריכים, ואז השוו שוב.',
                )}
              </small>
            </p>
          )}

          <div className="plan-scenario-grid">
            <ScenarioCard leg="a" title={pageText(locale, 'Scenario A', 'תרחיש A')} summary={payload.a} accent="accent-a" locale={locale} words={words} scopeText={scopeText} windowText={windowText} onAdopt={onAdopt} />
            <div className="plan-scenario-arrow" aria-hidden="true"><Flow size={18} /></div>
            <ScenarioCard leg="b" title={pageText(locale, 'Scenario B', 'תרחיש B')} summary={payload.b} accent="accent-b" locale={locale} words={words} scopeText={scopeText} windowText={windowText} onAdopt={onAdopt} />
          </div>

          <div className="plan-delta">
            <h3>{pageText(locale, 'What B does that A does not', 'מה B עושה ש-A לא')}</h3>
            <p className="plan-delta-window">{windowText}</p>
            <DeltaRow
              label={pageText(locale, 'Net after retention cost', 'נטו אחרי עלות שימור')}
              value={payload.delta?.revenue_net}
              locale={locale}
              formatter={formatCurrency}
              emphasis
            />
            <DeltaRow label={words.expectedRevenue} value={payload.delta?.gross} locale={locale} formatter={formatCurrency} />
            <DeltaRow label={words.retentionCost} value={payload.delta?.retention_cost} locale={locale} formatter={formatCurrency} />
            <DeltaRow label={pageText(locale, 'Retention', 'שימור')} value={payload.delta?.retention} locale={locale} formatter={formatNumber} suffix="pp" />
            <DeltaRow label={words.breaks} value={payload.delta?.breaks} locale={locale} formatter={formatNumber} />
            <DeltaRow label={pageText(locale, 'Ad seconds', 'שניות פרסום')} value={payload.delta?.ad_seconds} locale={locale} formatter={formatNumber} />
          </div>

          {week && days?.length ? (
            <CompareWeekTable
              locale={locale}
              dates={dates}
              days={days}
              running={false}
              elapsedMs={scope?.elapsed_ms}
              biggestDate={biggest?.date || null}
              onOpenDay={onOpenDay}
            />
          ) : null}

          {runCost ? <p className="plan-basis-note">{runCost}</p> : null}

          {payload.sameness?.identical && (
            <p className="plan-note plan-note-amber" role="status">
              {pageText(
                locale,
                'Both scenarios produced the same plan. At a fixed retention floor the engine settles on nearly the same schedule whatever the revenue weight, so the weight alone will not separate two scenarios. Change the retention floor, the hourly break cap, the caution or the engine focus to see the plan move.',
                'שני התרחישים הפיקו את אותה תוכנית. ברצפת צפייה קבועה המנוע מתכנס כמעט לאותו לוח בכל משקל הכנסה, ולכן המשקל לבדו לא יפריד בין שני תרחישים. שנו את רצפת הצפייה, את תקרת הברייקים לשעה, את הזהירות או את מיקוד המנוע כדי לראות את התוכנית זזה.',
              )}
              {payload.sameness.levers_that_differ?.length > 0 && (
                <span className="plan-note-detail">
                  {pageText(locale, 'Different between the two: ', 'שונה בין השניים: ')}
                  {payload.sameness.levers_that_differ.map((field) => leverLabel(field, locale)).join(', ')}
                </span>
              )}
            </p>
          )}

          <p className="plan-basis-note">
            {week
              ? pageText(
                locale,
                'Expected revenue is the optimizer’s own projection for every broadcast day in the plan’s own week, added up. Retention cost is the audience those breaks are modelled to lose, priced at the same rate. Net is one minus the other, on the same per-break basis the committed plan is priced on.',
                'ההכנסה הצפויה היא התחזית של האופטימייזר עצמו לכל ימי השידור בשבוע התוכנית עצמו, מסוכמים. עלות השימור היא הצופים שהברייקים האלה צפויים לאבד לפי המודל, מתומחרים באותו תעריף. הנטו הוא ההפרש ביניהם, על אותו בסיס לכל ברייק שבו מתומחרת התוכנית המחויבת.',
              )
              : pageText(
                locale,
                'Expected revenue is the optimizer’s own projection for this one broadcast day. Retention cost is the audience those breaks are modelled to lose, priced at the same rate. Net is one minus the other, on the same per-break basis the committed plan is priced on.',
                'ההכנסה הצפויה היא התחזית של האופטימייזר עצמו ליום השידור האחד הזה. עלות השימור היא הצופים שהברייקים האלה צפויים לאבד לפי המודל, מתומחרים באותו תעריף. הנטו הוא ההפרש ביניהם, על אותו בסיס לכל ברייק שבו מתומחרת התוכנית המחויבת.',
              )}
          </p>
          <p className="plan-basis-note">
            {week
              ? pageText(
                locale,
                'The blended score is the optimizer’s own balance of revenue against retention, normalised inside one broadcast day, so over a week it is the mean of its days and never a sum. It is not revenue minus retention cost; that subtraction is the net figure above it.',
                'הציון המשוקלל הוא האיזון של האופטימייזר בין הכנסה לשימור, מנורמל בתוך יום שידור אחד, ולכן על פני שבוע הוא ממוצע הימים ולעולם לא סכום. הוא אינו הכנסה פחות עלות שימור; החיסור הזה הוא ערך הנטו שמעליו.',
              )
              : pageText(
                locale,
                'The blended score is the optimizer’s own balance of revenue against retention. It is not revenue minus retention cost; that subtraction is the net figure above it.',
                'הציון המשוקלל הוא האיזון של האופטימייזר בין הכנסה לשימור. הוא אינו הכנסה פחות עלות שימור; החיסור הזה הוא ערך הנטו שמעליו.',
              )}
          </p>
        </>
      )}
    </section>
  );
}

export default ComparePanel;
