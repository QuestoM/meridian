import React from 'react';
import { Button } from '../../studio/actions';
import { Check, GitCompare, RefreshCcw, X } from 'lucide-react';
import { finiteNumber, formatCurrency, formatNumber, formatPercent, pageText } from '../../shell/format';
import { Figure, Name } from '../../shell/bidi';
import ScenarioLegForm from './ScenarioLegForm';
import CompareWeekTable from './CompareWeekTable';
import { leverLabel } from './plan-week-model';
import ScenarioAdopt from './ScenarioAdopt';
import { formatDay, formatSpan } from '../../shell/dates';

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
// only the floor, the caution and the engine focus move it. The hourly cap is a
// shared licence guardrail, not a scenario objective. And
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

function MatrixValue({ value, locale, formatter = formatNumber, suffix = '', signed = false, display }) {
  const number = finiteNumber(value);
  if (number === null) return <span className="is-unavailable">{'\u2014'}</span>;
  return <Figure>{signed && number > 0 ? '+' : ''}{display ?? formatter(number, locale)}{suffix}</Figure>;
}

function scenarioMeasures(summary, locale) {
  return {
    net: summary.revenue_net,
    objective: summary.objective,
    objectiveText: blendedScore(summary.objective, locale),
  };
}

function TradeoffTrace({ a, b, locale }) {
  const points = [
    { id: 'A', retention: finiteNumber(a?.average_retention), net: finiteNumber(a?.revenue_net) },
    { id: 'B', retention: finiteNumber(b?.average_retention), net: finiteNumber(b?.revenue_net) },
  ];
  if (points.some((point) => point.retention === null || point.net === null)) {
    return (
      <div className="plan-tradeoff-empty">
        {pageText(locale, 'The two measured outcomes do not expose both net and retention, so no trade-off trace is drawn.', 'שתי התוצאות שנמדדו אינן חושפות גם נטו וגם שימור, ולכן לא מצויר מסלול הכרעה.')}
      </div>
    );
  }
  const retentionValues = points.map((point) => point.retention);
  const netValues = points.map((point) => point.net);
  const retentionSpan = Math.max(0.0001, Math.max(...retentionValues) - Math.min(...retentionValues));
  const netSpan = Math.max(0.0001, Math.max(...netValues) - Math.min(...netValues));
  const mapped = points.map((point) => ({
    ...point,
    x: 28 + ((point.retention - Math.min(...retentionValues)) / retentionSpan) * 164,
    y: 116 - ((point.net - Math.min(...netValues)) / netSpan) * 88,
  }));
  // Equal measurements must occupy the same point. The tiny non-zero domain
  // above only prevents division by zero; this explicit branch keeps the plot
  // from implying a difference the server did not report.
  if (retentionValues[0] === retentionValues[1]) mapped.forEach((point) => { point.x = 110; });
  if (netValues[0] === netValues[1]) mapped.forEach((point) => { point.y = 72; });
  return (
    <figure className="plan-tradeoff" aria-label={pageText(locale, 'Two measured scenario outcomes: retention against net after retention cost', 'שתי תוצאות תרחיש שנמדדו: שימור מול נטו אחרי עלות שימור')}>
      <div className="plan-tradeoff-head">
        <strong>{pageText(locale, 'Outcome plot', 'גרף תוצאות')}</strong>
        <span>{pageText(locale, 'two measured outcomes · no interpolation', 'שתי תוצאות שנמדדו · ללא אינטרפולציה')}</span>
      </div>
      <svg viewBox="0 0 220 142" role="img" aria-hidden="true">
        <path className="plan-tradeoff-axis" d="M22 16V122H202" />
        <path className="plan-tradeoff-link" d={`M${mapped[0].x} ${mapped[0].y}L${mapped[1].x} ${mapped[1].y}`} />
        {mapped.map((point) => (
          <g key={point.id} className={`plan-tradeoff-point is-${point.id.toLowerCase()}`} transform={`translate(${point.x} ${point.y})`}>
            <circle r="7" />
            <text x="0" y="3">{point.id}</text>
          </g>
        ))}
      </svg>
      <div className="plan-tradeoff-labels">
        <span>{pageText(locale, 'Net ↑', 'נטו ↑')}</span>
        <span>{pageText(locale, 'Retention →', 'שימור ←')}</span>
      </div>
      <figcaption>
        {pageText(locale, 'The line only connects A and B; it does not claim outcomes between them.', 'הקו רק מחבר בין A ל־B; הוא אינו טוען לתוצאות ביניהן.')}
      </figcaption>
    </figure>
  );
}

function ComparisonInstrument({ payload, locale, words, windowText, scopeText, onAdopt }) {
  const a = payload.a || {};
  const b = payload.b || {};
  const delta = payload.delta || {};
  const measuredA = scenarioMeasures(a, locale);
  const measuredB = scenarioMeasures(b, locale);
  const netDelta = payload.delta?.revenue_net;
  const rows = [
    { key: 'net', label: pageText(locale, 'Net after retention cost', 'נטו אחרי עלות שימור'), a: measuredA.net, b: measuredB.net, d: netDelta, formatter: formatCurrency, headline: true },
    { key: 'gross', label: words.expectedRevenue, a: a.gross, b: b.gross, d: delta.gross, formatter: formatCurrency },
    { key: 'cost', label: words.retentionCost, a: a.retention_cost, b: b.retention_cost, d: delta.retention_cost, formatter: formatCurrency },
    { key: 'retention', label: pageText(locale, 'Average retention', 'שימור ממוצע'), a: a.average_retention, b: b.average_retention, d: delta.retention, formatter: formatPercent, deltaFormatter: formatNumber, deltaSuffix: 'pp' },
    { key: 'breaks', label: words.breaks, a: a.total_breaks, b: b.total_breaks, d: delta.breaks, formatter: formatNumber },
    { key: 'seconds', label: pageText(locale, 'Ad seconds', 'שניות פרסום'), a: a.total_ad_seconds, b: b.total_ad_seconds, d: delta.ad_seconds, formatter: formatNumber },
    { key: 'score', label: pageText(locale, 'Blended score', 'ציון משוקלל'), a: measuredA.objective, b: measuredB.objective, aDisplay: measuredA.objectiveText, bDisplay: measuredB.objectiveText, d: finiteNumber(b.objective) !== null && finiteNumber(a.objective) !== null ? Number(b.objective) - Number(a.objective) : null, formatter: blendedScore, deltaFormatter: blendedScore },
  ];
  return (
    <div className="plan-comparison-instrument">
      <div className="plan-comparison-matrix-wrap">
        <div className="plan-comparison-scope">
          <span>{windowText}</span>
          {scopeText ? <span>{scopeText}</span> : null}
        </div>
        <table className="plan-comparison-matrix">
          <thead>
            <tr>
              <th scope="col">{pageText(locale, 'Measured outcome', 'תוצאה שנמדדה')}</th>
              <th scope="col">A <span className={`plan-compliance${a.compliant ? ' ok' : ' warn'}`}>{a.compliant ? pageText(locale, 'within guardrails', 'בתוך המגבלות') : pageText(locale, 'breach', 'חריגה')}</span></th>
              <th scope="col">B <span className={`plan-compliance${b.compliant ? ' ok' : ' warn'}`}>{b.compliant ? pageText(locale, 'within guardrails', 'בתוך המגבלות') : pageText(locale, 'breach', 'חריגה')}</span></th>
              <th scope="col">B − A</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key} className={row.headline ? 'is-headline' : undefined}>
                <th scope="row">{row.label}</th>
                <td className="numeric"><MatrixValue value={row.a} locale={locale} formatter={row.formatter} display={row.aDisplay} /></td>
                <td className="numeric"><MatrixValue value={row.b} locale={locale} formatter={row.formatter} display={row.bDisplay} /></td>
                <td className="numeric"><MatrixValue value={row.d} locale={locale} formatter={row.deltaFormatter || row.formatter} suffix={row.deltaSuffix} signed /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <TradeoffTrace a={a} b={b} locale={locale} />
      <div className="plan-adopt-grid">
        <div><strong>{pageText(locale, 'Scenario A controls', 'בקרי תרחיש A')}</strong><ScenarioAdopt leg="a" summary={a} locale={locale} onAdopt={onAdopt} /></div>
        <div><strong>{pageText(locale, 'Scenario B controls', 'בקרי תרחיש B')}</strong><ScenarioAdopt leg="b" summary={b} locale={locale} onAdopt={onAdopt} /></div>
      </div>
      {(a.money_available !== true || b.money_available !== true) ? (
        <p className="plan-note plan-note-amber">
          {pageText(locale, 'At least one run did not expose money, so its monetary cells stay blank.', 'לפחות הרצה אחת לא חשפה נתוני כסף, ולכן התאים הכספיים שלה נשארים ריקים.')}
          {[a.money_reason, b.money_reason].filter(Boolean).map((reason) => <small className="plan-note-detail" key={reason}><Name>{reason}</Name></small>)}
        </p>
      ) : null}
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
  actionDisabled, actionDisabledReason, onLegChange, onCompare, onCancel, onAdopt, onOpenDay,
}) {
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
        <bdi>{week ? formatSpan(scope.date_from, scope.date_to, locale) : formatDay(scope.day)}</bdi>
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
    <section className="card plan-section" aria-labelledby="plan-compare-title">
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
          <Button className="run-button" type="button" variant="contained" disabled={running || actionDisabled} title={actionDisabledReason || undefined} onClick={onCompare}>
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

          <ComparisonInstrument payload={payload} locale={locale} words={words} scopeText={scopeText} windowText={windowText} onAdopt={onAdopt} />

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
                'Both scenarios produced the same plan. At a fixed retention floor the engine settles on nearly the same schedule whatever the revenue weight, so the weight alone will not separate two scenarios. Change the retention floor, the caution or the engine focus to see the plan move.',
                'שני התרחישים הפיקו את אותה תוכנית. ברצפת צפייה קבועה המנוע מתכנס כמעט לאותו לוח בכל משקל הכנסה, ולכן המשקל לבדו לא יפריד בין שני תרחישים. שנו את רצפת הצפייה, את הזהירות או את מיקוד המנוע כדי לראות את התוכנית זזה.',
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
