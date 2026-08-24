import React, { useRef, useState } from 'react';
import { Button } from '../studio/actions';
import { Numeric, finiteNumber, formatCurrency, formatNumber, formatPercent, pageText } from '../shell/format';
import { useAddressParam } from '../shell/address-state';
import { formatDay, formatSpan } from '../shell/dates';
import { programTypeLabel, recommendationTitle } from '../shell/labels';
import { Name, isolate } from '../shell/bidi';
import TodayDecisionDetail from './TodayDecisionDetail';

// Answer three: what needs a decision.
//
// The same five rows the product has always produced, with the same figures,
// and one line the product never printed: what they are ranked by. A list
// called priority with no stated order is a claim, not an answer.
//
// A row opens the decision on this screen, because that is the promise the
// line above the list makes and this is the surface that can keep it. The
// optimizer stays one click further on, from inside the opened row, so the
// path the product already had is still there for the person who wants to act
// rather than to read.
//
// The list also prints the span it was drawn from, in the same fact-list form
// the money answer above it uses. It has to: the ranking scans the whole saved
// plan, so its rows are routinely dated outside the seven-day window the money
// answer names, and two spans on one screen with only one of them stated reads
// as an error in the product rather than a fact about the plan.

const RISK_WORDS = {
  High: ['Below the retention floor', 'מתחת לרצפת השימור'],
  Medium: ['Close to the retention floor', 'קרוב לרצפת השימור'],
  Low: ['Clear of the retention floor', 'מעל רצפת השימור'],
};

// The span, in the words the money panel's own scope line uses, so the two
// spans on this screen are read against each other rather than in two
// vocabularies. The last fact is the one that resolves the apparent
// contradiction. It is claimed only when the two spans differ, and it is safe
// to phrase as reaching past the window because the window is always a slice
// of this same plan: the endpoint cuts it out of the summary this span is read
// from, so the plan can never be the narrower of the two.
function scopeLine(today, locale) {
  const scope = (today.decisions || {}).scope || {};
  if (!scope.date_from || !scope.date_to) return '';
  const moneyScope = ((today.money || {}).scope) || {};
  const days = finiteNumber(scope.n_dates);
  const span = formatSpan(scope.date_from, scope.date_to, locale);
  const beyondTheWindow = scope.date_from !== moneyScope.date_from || scope.date_to !== moneyScope.date_to;
  const parts = [];
  if (scope.channel) parts.push(isolate(scope.channel));
  parts.push(days ? pageText(locale, `${span} (${formatNumber(days, locale)} days)`, `${span} (${formatNumber(days, locale)} ימים)`) : span);
  if (scope.inclusive) parts.push(pageText(locale, 'both dates included', 'שני התאריכים כלולים'));
  if (beyondTheWindow) parts.push(pageText(locale, 'the whole saved plan, not only the window above', 'כל התוכנית השמורה, לא רק החלון שלמעלה'));
  else parts.push(pageText(locale, 'from the saved plan', 'מהתוכנית השמורה'));
  return parts.join(' · ');
}

export function TodayDecisions({ today, locale, onOpenInOptimizer, onOpenSettings }) {
  // The open decision is an address (decision in shell/nav.js): Back reopens
  // the decision the operator was weighing rather than collapsing the queue.
  const [openId, setOpenId] = useAddressParam('decision', '');
  // Where the keyboard came from, so closing returns it there rather than
  // dropping it at the top of the document.
  const rows = useRef({});
  const decisions = today.decisions || {};
  const items = Array.isArray(decisions.items) ? decisions.items : [];
  const withheld = decisions.unavailable;
  const scope = scopeLine(today, locale);

  if (withheld) {
    return (
      <section className="page-panel today-answer today-answer-decisions" aria-label={pageText(locale, 'What needs a decision', 'מה דורש החלטה')}>
        <div className="panel-head">
          <h2>{pageText(locale, 'Operator review queue', 'החלטות שממתינות לבדיקה')}</h2>
        </div>
        <p className="today-basis">{pageText(locale, withheld.reason_en, withheld.reason_he)}</p>
        <div className="today-target-actions">
          <Button className="today-primary" type="button" variant="contained" onClick={() => onOpenSettings && onOpenSettings()}>
            {pageText(locale, withheld.needs_en, withheld.needs_he)}
          </Button>
        </div>
      </section>
    );
  }

  return (
    <section className="page-panel today-answer today-answer-decisions" aria-label={pageText(locale, 'What needs a decision', 'מה דורש החלטה')}>
      <div className="panel-head">
        <h2>{pageText(locale, 'Operator review queue', 'החלטות שממתינות לבדיקה')}</h2>
        <span><Numeric>{formatNumber(items.length, locale)}</Numeric> {pageText(locale, 'items', 'פריטים')}</span>
      </div>
      <p className="today-basis today-decision-scope">
        {[
          scope,
          pageText(locale, 'Highest expected revenue first; open a row for the retention evidence and plan decision.', 'מסודר לפי הכנסה צפויה; פתיחת שורה מציגה את נתוני השימור ואת החלטת התכנון.'),
        ].filter(Boolean).join(' · ')}
      </p>
      <div className="decision-list today-decision-list">
        {items.slice(0, 5).map((item) => {
          const risk = RISK_WORDS[item.risk] || RISK_WORDS.Low;
          const open = openId === item.id;
          const panelId = `today-decision-${item.id}`;
          const close = () => {
            setOpenId('');
            const row = rows.current[item.id];
            if (row && row.focus) row.focus();
          };
          return (
            <React.Fragment key={item.id || item.title}>
              <Button
                className={`decision-row${open ? ' open' : ''}`}
                type="button"
                aria-expanded={open}
                aria-controls={open ? panelId : undefined}
                ref={(node) => { rows.current[item.id] = node; }}
                onClick={() => (open ? close() : setOpenId(String(item.id || '')))}
              >
                <div>
                  <strong>{recommendationTitle(item, locale)}</strong>
                  <Name>
                    {[
                      programTypeLabel(item.program_type, locale) || pageText(locale, 'Mixed', 'מעורב'),
                      item.date ? formatDay(item.date) : '',
                      pageText(locale, risk[0], risk[1]),
                    ].filter(Boolean).join(' · ')}
                  </Name>
                </div>
                <div>
                  <strong><Numeric>{formatCurrency(item.impact, locale)}</Numeric></strong>
                  <span><Numeric>{formatPercent(item.retention, locale)}</Numeric></span>
                </div>
              </Button>
              {open ? (
                <TodayDecisionDetail
                  panelId={panelId}
                  item={item}
                  locale={locale}
                  onClose={close}
                  onOpenInOptimizer={onOpenInOptimizer}
                />
              ) : null}
            </React.Fragment>
          );
        })}
      </div>
    </section>
  );
}

export default TodayDecisions;
