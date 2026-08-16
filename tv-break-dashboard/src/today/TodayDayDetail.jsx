import React, { useEffect, useRef, useState } from 'react';
import { Button } from '../studio/actions';
import { ChevronDown, ChevronUp, Download, X } from 'lucide-react';
import { Numeric, finiteNumber, formatCurrency, formatNumber, pageText } from '../shell/format';
import { formatDay } from '../shell/dates';
import { programTypeLabel } from '../shell/labels';
import { fetchTodayDay } from './today-data';
import { SEGMENT_COLUMNS, download, scopeComment, toCsv } from './today-export';

// The second level of the drill: one day, and the plan rows that produced it.
//
// It opens in place under the day it belongs to, it says where it sits in the
// set it came from, and the two arrows walk that set without going back up. The
// level below this one is named rather than faked: what each break delivered
// needs a delivery feed, and the payload says so in the reader's own language.
//
// The control at the end of the head opens the plan and says exactly that. It
// cannot open this day: the shell renders the plan surface from a frozen router
// and that surface takes no date, so landing on one day is a contract the plan
// piece has to publish before anything here can carry one. Until it does, the
// button names the place it reaches rather than the day it does not.

export function TodayDayDetail({ date, scope, locale, onClose, onWalk, onOpenPlan }) {
  const [state, setState] = useState({ status: 'loading', day: null, error: '' });
  const panel = useRef(null);

  useEffect(() => {
    let active = true;
    setState({ status: 'loading', day: null, error: '' });
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
  }, [date]);

  const day = state.day;
  const position = (day && day.position) || {};

  function walk(next) {
    if (next && onWalk) onWalk(next);
  }

  function keyDown(event) {
    if (event.key === 'Escape') {
      event.preventDefault();
      if (onClose) onClose();
      return;
    }
    if (event.key === 'ArrowDown' || event.key === 'j') {
      event.preventDefault();
      walk(position.next);
      return;
    }
    if (event.key === 'ArrowUp' || event.key === 'k') {
      event.preventDefault();
      walk(position.previous);
    }
  }

  function exportRows() {
    if (!day || !day.rows) return;
    const text = [
      scopeComment({ ...scope, date_from: day.date, date_to: day.date }, [
        ['grain', 'plan segment'],
        ['rows', day.row_count],
        ['day_total_ils', day.projected_revenue],
      ]),
      '',
      toCsv(SEGMENT_COLUMNS, day.rows),
    ].join('\n');
    download(`meridian-${day.date}-segments.csv`, text);
  }

  return (
    <section
      className="today-day-detail"
      ref={panel}
      tabIndex={-1}
      onKeyDown={keyDown}
      aria-label={pageText(locale, `The rows behind ${formatDay(date)}`, `השורות שמאחורי ${formatDay(date)}`)}
    >
      <div className="today-day-detail-head">
        <strong><Numeric>{formatDay(date)}</Numeric></strong>
        <span className="today-day-position">
          <Numeric>{`${position.index || '-'} / ${position.total || '-'}`}</Numeric>
        </span>
        <Button className="today-icon-button" type="button" disabled={!position.previous} onClick={() => walk(position.previous)} aria-label={pageText(locale, 'Previous day', 'היום הקודם')}>
          <ChevronUp size={15} />
        </Button>
        <Button className="today-icon-button" type="button" disabled={!position.next} onClick={() => walk(position.next)} aria-label={pageText(locale, 'Next day', 'היום הבא')}>
          <ChevronDown size={15} />
        </Button>
        <span className="today-key-hint">{pageText(locale, 'J and K walk the week, Esc closes', 'J ו־K מדלגים בין ימי השבוע, Esc סוגר')}</span>
        <Button className="today-link-button" type="button" onClick={() => onOpenPlan && onOpenPlan()}>
          {pageText(locale, 'Open the plan', 'פתחו את התוכנית')}
        </Button>
        <Button className="today-icon-button" type="button" onClick={exportRows} disabled={state.status !== 'ready'} aria-label={pageText(locale, 'Download these rows', 'הורדת השורות האלה')}>
          <Download size={15} />
        </Button>
        <Button className="today-icon-button" type="button" onClick={onClose} aria-label={pageText(locale, 'Close', 'סגירה')}>
          <X size={15} />
        </Button>
      </div>

      {state.status === 'loading' ? (
        <p className="today-note">{pageText(locale, 'Reading the rows behind this day.', 'קורא את השורות שמאחורי היום הזה.')}</p>
      ) : null}
      {state.status === 'error' ? (
        <p className="today-note">{pageText(locale, `These rows did not answer. ${state.error}`, `השורות האלה לא נענו. ${state.error}`)}</p>
      ) : null}

      {day && day.available ? <DayRowTable day={day} locale={locale} /> : null}
      {day && !day.available ? (
        <p className="today-note">{pageText(locale, `No rows behind this day: ${day.reason}.`, 'אין שורות מאחורי היום הזה.')}</p>
      ) : null}
      {day ? (
        <p className="today-note today-delivered-note">
          {pageText(locale, day.delivered.reason_en, day.delivered.reason_he)}
          {' '}
          {pageText(locale, day.delivered.needs_en, day.delivered.needs_he)}
        </p>
      ) : null}
    </section>
  );
}

function DayRowTable({ day, locale }) {
  const rows = Array.isArray(day.rows) ? day.rows : [];
  return (
    <>
      <p className="today-note">
        {pageText(
          locale,
          `${formatNumber(day.row_count, 'en')} plan rows, ${formatNumber(day.total_breaks, 'en')} breaks, ${formatNumber(day.total_ad_seconds, 'en')} ad seconds, highest expected revenue first.`,
          `${formatNumber(day.row_count, 'he')} שורות תוכנית, ${formatNumber(day.total_breaks, 'he')} ברייקים, ${formatNumber(day.total_ad_seconds, 'he')} שניות פרסום, מההכנסה הצפויה הגבוהה לנמוכה.`,
        )}
      </p>
      <div className="today-segment-rows" role="list">
        {rows.map((row) => (
          <div className="today-segment-row" role="listitem" key={row.segment_id || `${row.start_clock}-${row.projected_revenue}`}>
            <span className="today-segment-clock"><Numeric>{row.start_clock}</Numeric></span>
            <span className="today-segment-type">{programTypeLabel(row.program_type, locale) || row.program_type}</span>
            <span className="today-segment-breaks">
              <Numeric>{pageText(locale, `${formatNumber(row.breaks, locale)} × ${formatNumber(row.ad_seconds, locale)}s`, `${formatNumber(row.breaks, locale)} × ${formatNumber(row.ad_seconds, locale)} שנ'`)}</Numeric>
            </span>
            <span className="today-segment-retention"><Numeric>{`${formatNumber(row.retention_percent, locale)}%`}</Numeric></span>
            <span className="today-segment-money"><Numeric>{formatCurrency(row.projected_revenue, locale)}</Numeric></span>
            <span className="today-segment-share">
              <Numeric>{finiteNumber(row.share_percent) === null ? '-' : `${formatNumber(row.share_percent, locale)}%`}</Numeric>
            </span>
          </div>
        ))}
      </div>
      <p className="today-note">
        {day.reconciled
          ? pageText(locale, 'These rows sum to this day, to the shekel.', 'סכום השורות האלה שווה ליום הזה, עד לשקל.')
          : pageText(locale, `These rows and this day differ by ${formatCurrency(day.residual_ils, locale)}.`, `בין השורות האלה ליום הזה יש הפרש של ${formatCurrency(day.residual_ils, locale)}.`)}
      </p>
    </>
  );
}

export default TodayDayDetail;
