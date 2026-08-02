import React, { useEffect, useMemo, useState } from 'react';
import { Star } from 'lucide-react';
import { formatNumber, formatPercent, pageText } from '../../shell/format';
import DayPicker from '../day/DayPicker';
import BreakInspector from './BreakInspector';
import { clockOf, exactCurrency } from '../day/day-board-model';
import { fetchDay, fetchDays } from '../day/day-board-actions';
import { basisSentence, roundingSentence, shareOfDay, sumRevenue, visibleRows } from './break-board-model';
import './break-board.css';

// Plan, at break zoom: every break in one broadcast day, as objects.
//
// This is the money layer made visible. Each row is one addressable break with
// the revenue the optimizer credited to it. Every break in the day, added up, is
// the day: measured on רשת 13 / 2024-11-01, the eighty rows sum to 1,062,669.90
// against a day of 1,062,669.88, a two agora gap that is the per-row rounding
// the route serves and nothing else, and both print as 1,062,670. Delivered is a
// state and never a figure, because the plan and the one daily spot ledger
// overlap on zero dates.
//
// Every row opens. A break is a real object with its own address, so a name, a
// figure and a badge on this board all resolve to the same drawer.
//
// The footer adds up the rows that are on screen, and it is written that way
// because it once did not. Measured on רשת 13 / 2024-11-01 with one break marked
// gold: the gold filter left three rows worth 10,712, 10,163 and 9,614, and the
// line under them printed 1,028,206 ILS, the whole day, 33.7 times the column it
// claimed to total, under the label "sum over these breaks". With no gold break
// in the day the same filter left no rows, no empty state and the same figure.
// So the total is now the sum of what is displayed, the day's own total is a
// second line that appears when a filter is on, and an emptied filter says which
// mark is missing and where it is made.
function BreakBoard({ locale, notify }) {
  const he = locale === 'he';
  const label = (en, hebrew) => (he ? hebrew : en);
  const [days, setDays] = useState(null);
  const [day, setDay] = useState('');
  const [board, setBoard] = useState(null);
  const [error, setError] = useState('');
  const [open, setOpen] = useState(null);
  const [goldOnly, setGoldOnly] = useState(false);

  useEffect(() => {
    let alive = true;
    fetchDays()
      .then((payload) => {
        if (!alive) return;
        setDays(payload);
        setDay((current) => current || (payload.days && payload.days.length ? payload.days[0] : ''));
      })
      .catch((fetchError) => { if (alive) setError(fetchError.message); });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (!day) return undefined;
    let alive = true;
    setBoard(null);
    fetchDay(day)
      .then((payload) => { if (alive) setBoard(payload); })
      .catch((fetchError) => { if (alive) setError(fetchError.message); });
    return () => { alive = false; };
  }, [day]);

  const all = board ? board.breaks : [];
  // The rows the table draws and the rows the footer adds are the one list, and
  // the figure under the column is that list summed. Never the day, unless the
  // day is what is on screen.
  const rows = useMemo(() => visibleRows(all, goldOnly), [all, goldOnly]);
  const shown = useMemo(() => sumRevenue(rows), [rows]);
  const dayRevenue = board ? Number(board.totals.revenue) : 0;
  const share = shareOfDay(shown, dayRevenue);

  if (error) {
    return (
      <section className="page-panel">
        <p className="break-board-error" dir="auto">{error}</p>
      </section>
    );
  }

  if (days && !days.available) {
    return (
      <section className="page-panel">
        <h2>{pageText(locale, 'No breaks to open yet', 'אין עדיין ברייקים לפתיחה')}</h2>
        <p dir="auto">{(he && days.reason_he) || days.reason}</p>
      </section>
    );
  }

  return (
    <section className="page-panel break-board" dir={he ? 'rtl' : 'ltr'}>
      <div className="panel-head">
        <h2>{pageText(locale, 'Breaks in the day', 'ברייקים ביום')}</h2>
        <div className="panel-head-tools">
          <button
            type="button"
            className={goldOnly ? 'break-filter is-on' : 'break-filter'}
            aria-pressed={goldOnly}
            onClick={() => setGoldOnly((current) => !current)}
          >
            <Star size={12} aria-hidden="true" />
            {pageText(locale, 'Gold breaks', 'ברייקי זהב')}
          </button>
          <span dir="ltr">{goldOnly ? `${rows.length} / ${all.length}` : rows.length}</span>
        </div>
      </div>
      <DayPicker
        days={days ? days.days : []}
        value={day}
        onChange={setDay}
        locale={locale}
        channel={days ? days.operator_channel : ''}
      />
      {!board && <p>{label('Opening the day', 'פותח את היום')}</p>}
      {board && rows.length === 0 && (
        <EmptyBoard board={board} goldOnly={goldOnly} locale={locale} onClearFilter={() => setGoldOnly(false)} />
      )}
      {board && rows.length > 0 && (
        <>
          <table className="break-table">
            <thead>
              <tr>
                <th scope="col">{pageText(locale, 'Start', 'התחלה')}</th>
                <th scope="col">{pageText(locale, 'Programme', 'תוכנית')}</th>
                <th scope="col">{pageText(locale, 'Order', 'סדר')}</th>
                <th scope="col">{pageText(locale, 'Length', 'אורך')}</th>
                <th scope="col">{pageText(locale, 'Expected revenue', 'הכנסה צפויה')}</th>
                <th scope="col">{label('Delivered', 'שסופק בפועל')}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.break_id} onClick={() => setOpen(row.break_id)} tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') setOpen(row.break_id);
                  }}
                >
                  <td dir="ltr">{clockOf(row.start_seconds)}</td>
                  <td dir="auto">
                    {row.is_gold && <Star size={11} aria-hidden="true" />}
                    {row.programme}
                  </td>
                  <td dir="ltr">{row.ordinal} / {row.breaks_in_segment}</td>
                  <td dir="ltr">{formatNumber(row.duration_seconds, locale)}s</td>
                  <td dir="ltr">{exactCurrency(row.projected_revenue, locale)}</td>
                  <td className="break-cell-muted" dir="auto">{(he && row.delivered.reason_he) || row.delivered.reason}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <td colSpan={4}>
                  {label('Sum over these breaks', 'סכום על הברייקים האלה')}
                  <span className="break-foot-count" dir="ltr">{rows.length} / {all.length}</span>
                </td>
                <td dir="ltr">{exactCurrency(shown, locale)}</td>
                <td className="break-cell-muted" dir="auto">{board.basis.channel}, {board.basis.day}</td>
              </tr>
              {goldOnly && (
                <tr className="break-foot-day">
                  <td colSpan={4}>{label('The whole day, every break', 'כל היום, כל הברייקים')}</td>
                  <td dir="ltr">{exactCurrency(dayRevenue, locale)}</td>
                  <td className="break-cell-muted" dir="auto">{board.basis.channel}, {board.basis.day}</td>
                </tr>
              )}
            </tfoot>
          </table>
          <p className="break-board-basis" dir="auto">{basisSentence({ goldOnly, shownCount: rows.length, total: all.length, portion: share === null ? null : formatPercent(share, locale), locale })}</p>
          <p className="break-board-rounding" dir="auto">{roundingSentence(locale)}</p>
        </>
      )}
      {open && (
        <BreakInspector
          breakId={open}
          locale={locale}
          siblings={rows.map((row) => row.break_id)}
          onNavigate={setOpen}
          onClose={() => setOpen(null)}
          notify={notify}
        />
      )}
    </section>
  );
}

// The filter emptied the table, so the table is replaced by what is missing.
//
// It names the mark rather than the absence of rows, gives the exact place the
// mark is made and the ceiling the guardrails put on it, keeps the day's own
// money on screen with its own label so nothing is hidden by the filter, and
// offers the way back. Every figure here is served: the break count, the day's
// revenue, the gold ceiling and whether gold is switched on at all.
export function EmptyBoard({ board, goldOnly, locale, onClearFilter }) {
  const he = locale === 'he';
  const label = (en, hebrew) => (he ? hebrew : en);
  const gold = board.gold || {};
  const goldOff = gold.enabled === false;
  return (
    <div className="break-board-empty">
      <h3>{goldOnly
        ? label('No break in this day is marked gold', 'אין ביום הזה ברייק שמסומן כברייק זהב')
        : label('The plan places no break in this day', 'התוכנית לא מציבה ברייקים ביום הזה')}
      </h3>
      {goldOnly && goldOff && (
        <p dir="auto">{label('Gold breaks are switched off in settings, so no break here can carry the mark. Switch them on in settings first.', 'ברייקי זהב כבויים בהגדרות, ולכן אף ברייק כאן אינו יכול לשאת את הסימון. הפעילו אותם בהגדרות תחילה.')}</p>
      )}
      {goldOnly && !goldOff && (
        <p dir="auto">{label(`Marking is done on the day board: open Plan, the day, select a break and press G, or use its Gold break button. Up to ${formatNumber(gold.max_per_day, locale)} gold breaks a day are allowed on this channel.`, `הסימון נעשה בלוח היום: פתחו את תוכנית, היום, בחרו ברייק והקישו G, או השתמשו בכפתור ברייק זהב שלו. מותרים בערוץ הזה עד ${formatNumber(gold.max_per_day, locale)} ברייקי זהב ביום.`)}</p>
      )}
      {!goldOnly && (
        <p dir="auto">{label('Every programme on this day was planned with zero breaks, so there is nothing to open at break zoom.', 'כל רצועות השידור ביום הזה תוכננו ללא ברייקים, ולכן אין מה לפתוח בזום הברייק.')}</p>
      )}
      <p className="break-board-empty-day" dir="auto">
        {label(`The day itself holds ${formatNumber(board.totals.breaks, locale)} breaks worth ${exactCurrency(board.totals.revenue, locale)}`, `היום עצמו מחזיק ${formatNumber(board.totals.breaks, locale)} ברייקים בשווי ${exactCurrency(board.totals.revenue, locale)}`)}
        <span className="break-cell-muted" dir="auto">{board.basis.channel}, {board.basis.day}</span>
      </p>
      {goldOnly && (
        <button type="button" className="break-filter" onClick={onClearFilter}>
          {label('Show every break in the day', 'הצגת כל הברייקים ביום')}
        </button>
      )}
    </div>
  );
}

export default BreakBoard;
