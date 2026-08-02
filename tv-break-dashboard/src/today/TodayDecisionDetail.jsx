import React, { useEffect, useRef, useState } from 'react';
import { Button } from '@mui/material';
import { X } from 'lucide-react';
import { Numeric, finiteNumber, formatCurrency, formatNumber, formatPlanDate, pageText } from '../shell/format';
import { programTypeLabel } from '../shell/labels';
import { fetchTodayDay } from './today-data';

// The third answer, opened where it was clicked.
//
// A list called priority decisions has to open the decision, and the decision
// is not a page somewhere else: it is one segment of the saved plan, the two
// figures it was ranked on, the reason it is on the list, and the change it
// implies. All four are in hand here, so the row opens them here, in place,
// with no navigation and no wait for another surface to select anything.
//
// The fifth thing, the plan row itself, is fetched from the same drill the
// money figure uses, so the decision is checked against the plan rather than
// asserted: the segment appears among that day's rows, carrying the same
// revenue and the same retention the row above printed, and its share of the
// day says how large it is. Where that read cannot answer, this panel says so
// and prints nothing in its place.

// What the plan implies for this segment, in the operator's own words. These
// are proposals a person still has to make; nothing here performs one.
const PROPOSALS = {
  gold: ['Mark it as a gold break', 'לסמן אותו כברייק זהב'],
  pin: ['Pin the number of breaks it carries', 'לנעוץ את מספר הברייקים שהוא נושא'],
  lower_count: ['Lower the number of breaks it carries', 'להפחית את מספר הברייקים שהוא נושא'],
  forbid: ['Place no break in it', 'לא לשבץ בו ברייק'],
};

function matchedRow(day, item) {
  const rows = day && Array.isArray(day.rows) ? day.rows : [];
  const segment = String(item.segment_id || '').trim();
  if (segment) return rows.find((row) => String(row.segment_id || '').trim() === segment) || null;
  const clock = String(item.start_clock || '').trim();
  return clock ? rows.find((row) => String(row.start_clock || '').trim() === clock) || null : null;
}

export function TodayDecisionDetail({ panelId, item, locale, onClose, onOpenInOptimizer }) {
  const [state, setState] = useState({ status: 'loading', day: null, error: '' });
  const panel = useRef(null);
  const date = String(item.date || '').trim();

  useEffect(() => {
    let active = true;
    setState({ status: 'loading', day: null, error: '' });
    if (!date) {
      setState({ status: 'ready', day: null, error: '' });
      return undefined;
    }
    fetchTodayDay(date)
      .then((day) => {
        if (active) setState({ status: 'ready', day, error: '' });
      })
      .catch((error) => {
        if (active) setState({ status: 'error', day: null, error: String(error.message || error) });
      });
    return () => {
      active = false;
    };
  }, [date]);

  useEffect(() => {
    if (panel.current && panel.current.focus) panel.current.focus();
  }, [item.id]);

  function keyDown(event) {
    if (event.key === 'Escape') {
      event.preventDefault();
      if (onClose) onClose();
    }
  }

  const proposal = PROPOSALS[String(item.proposed_kind || '')];
  return (
    <section
      className="today-decision-detail"
      id={panelId}
      ref={panel}
      tabIndex={-1}
      onKeyDown={keyDown}
      aria-label={pageText(locale, 'What this decision is made of', 'ממה מורכבת ההחלטה הזו')}
    >
      <div className="today-decision-detail-head">
        <strong>{pageText(locale, 'What this decision is made of', 'ממה מורכבת ההחלטה הזו')}</strong>
        <span className="today-key-hint">{pageText(locale, 'Esc closes', 'Esc סוגר')}</span>
        <Button className="today-icon-button" type="button" onClick={onClose} aria-label={pageText(locale, 'Close', 'סגירה')}>
          <X size={15} />
        </Button>
      </div>

      <p className="today-note">
        {pageText(locale, `Why it is on the list: ${item.rationale || ''}`, `למה היא ברשימה: ${item.rationale_he || ''}`)}
      </p>
      <p className="today-note">
        {proposal
          ? pageText(locale, `What it proposes: ${proposal[0]}`, `מה היא מציעה: ${proposal[1]}`)
          : pageText(locale, 'There is nothing concrete to change on this segment, so this row is advisory only.', 'אין במקטע הזה שינוי מוחשי לבצע, ולכן השורה הזו מייעצת בלבד.')}
      </p>

      <DecisionRows state={state} item={item} date={date} locale={locale} />

      <div className="today-target-actions">
        <Button className="today-secondary" type="button" onClick={() => onOpenInOptimizer && onOpenInOptimizer(item)}>
          {pageText(locale, 'Open the optimizer', 'פתחו את האופטימייזר')}
        </Button>
      </div>
    </section>
  );
}

function DecisionRows({ state, item, date, locale }) {
  if (state.status === 'loading') {
    return <p className="today-note">{pageText(locale, 'Reading the plan rows behind this decision.', 'קורא את שורות התוכנית שמאחורי ההחלטה הזו.')}</p>;
  }
  if (state.status === 'error') {
    return <p className="today-note">{pageText(locale, `These rows did not answer. ${state.error}`, `השורות האלה לא נענו. ${state.error}`)}</p>;
  }
  const day = state.day;
  if (!day || !day.available) {
    return <p className="today-note">{pageText(locale, `No plan rows behind this day: ${(day && day.reason) || 'this decision carries no date'}.`, 'אין שורות תוכנית מאחורי היום הזה.')}</p>;
  }
  const row = matchedRow(day, item);
  if (!row) {
    return <p className="today-note">{pageText(locale, "That day's plan rows carry no row for this segment.", 'בשורות התוכנית של אותו יום אין שורה למקטע הזה.')}</p>;
  }
  const share = finiteNumber(row.share_percent);
  const printedDate = formatPlanDate(date, locale);
  return (
    <>
      <p className="today-note">
        {pageText(
          locale,
          `One of ${formatNumber(day.row_count, 'en')} plan rows on ${printedDate}, which together are ${formatCurrency(day.projected_revenue, 'en')}.`,
          `אחת מתוך ${formatNumber(day.row_count, 'he')} שורות תוכנית ב${printedDate}, שסכומן ${formatCurrency(day.projected_revenue, 'he')}.`,
        )}
      </p>
      <div className="today-segment-rows" role="list">
        <div className="today-segment-row" role="listitem">
          <span className="today-segment-clock"><Numeric>{row.start_clock}</Numeric></span>
          <span className="today-segment-type">{programTypeLabel(row.program_type, locale) || row.program_type}</span>
          <span className="today-segment-breaks">
            <Numeric>{`${formatNumber(row.breaks, locale)} × ${formatNumber(row.ad_seconds, locale)}s`}</Numeric>
          </span>
          <span className="today-segment-retention"><Numeric>{`${formatNumber(row.retention_percent, locale)}%`}</Numeric></span>
          <span className="today-segment-money"><Numeric>{formatCurrency(row.projected_revenue, locale)}</Numeric></span>
          <span className="today-segment-share">
            <Numeric>{share === null ? '-' : `${formatNumber(share, locale)}%`}</Numeric>
          </span>
        </div>
      </div>
      <p className="today-note">
        {share === null
          ? pageText(locale, 'The share this row is of that day could not be computed.', 'לא ניתן לחשב איזה חלק מהיום הזה מהווה השורה.')
          : pageText(locale, `This row is ${formatNumber(share, 'en')}% of that day.`, `השורה הזו היא ${formatNumber(share, 'he')}% מהיום הזה.`)}
      </p>
      <AgreementLine row={row} item={item} locale={locale} />
    </>
  );
}

// Whether the plan row and the row above it carry the same two figures. It is
// checked rather than claimed: two reads of one segment agreeing is the whole
// point of opening it, and if they ever stop agreeing the difference is what a
// reader needs, not a sentence saying they match.
function AgreementLine({ row, item, locale }) {
  const money = finiteNumber(row.projected_revenue);
  const retention = finiteNumber(row.retention_percent);
  const listedMoney = finiteNumber(item.impact);
  const listedRetention = finiteNumber(item.retention);
  if (money === null || retention === null || listedMoney === null || listedRetention === null) {
    return <p className="today-note">{pageText(locale, 'One of the two figures is missing on one side, so they cannot be compared.', 'אחד משני המספרים חסר באחד הצדדים, ולכן אי אפשר להשוות ביניהם.')}</p>;
  }
  if (money === listedMoney && retention === listedRetention) {
    return <p className="today-note">{pageText(locale, 'It carries the same revenue and the same retention the row above prints.', 'היא נושאת את אותה הכנסה ואת אותו שימור שמופיעים בשורה שמעל.')}</p>;
  }
  return (
    <p className="today-note">
      {pageText(
        locale,
        `It reads ${formatCurrency(money, 'en')} and ${formatNumber(retention, 'en')}% here against ${formatCurrency(listedMoney, 'en')} and ${formatNumber(listedRetention, 'en')}% in the row above.`,
        `כאן היא נקראת ${formatCurrency(money, 'he')} ו־${formatNumber(retention, 'he')}%, ובשורה שמעל ${formatCurrency(listedMoney, 'he')} ו־${formatNumber(listedRetention, 'he')}%.`,
      )}
    </p>
  );
}

export default TodayDecisionDetail;
