import React, { useMemo, useState } from 'react';
import { ArrowUpRight, CircleAlert, Radio } from 'lucide-react';
import { Numeric, finiteNumber, formatCurrency, formatNumber, formatPercent, pageText } from '../shell/format';
import { formatDay, formatSpan } from '../shell/dates';
import { Button } from '../studio/actions';
import { Pressable } from '../studio/dom-controls';
import { Code, Name } from '../shell/bidi';
import './transmission-ribbon.css';

const MINUTE = 60;
const HOUR = 60 * MINUTE;
const MAX_PROGRAMS = 8;

function asSeconds(value) {
  const number = finiteNumber(value);
  return number === null ? null : number;
}

function displayPrograms(schedule) {
  const source = Array.isArray(schedule?.break_operations?.programs)
    ? schedule.break_operations.programs
    : [];
  const dated = source.filter((program) => program?.date && asSeconds(program.start_seconds) !== null && asSeconds(program.end_seconds) !== null);
  if (!dated.length) return [];
  const date = dated[0].date;
  const sameDay = dated
    .filter((program) => program.date === date && Number(program.duration_minutes || 0) >= 8)
    .sort((a, b) => Number(a.start_seconds) - Number(b.start_seconds));
  const sequential = [];
  let previousEnd = -1;
  for (const program of sameDay) {
    const start = Number(program.start_seconds);
    if (start < previousEnd - MINUTE) continue;
    sequential.push(program);
    previousEnd = Number(program.end_seconds);
    if (sequential.length === MAX_PROGRAMS) break;
  }
  return sequential;
}

function rangeFor(programs) {
  if (!programs.length) return null;
  const first = Math.floor(Number(programs[0].start_seconds) / HOUR) * HOUR;
  const last = Math.ceil(Number(programs[programs.length - 1].end_seconds) / HOUR) * HOUR;
  return { start: first, end: Math.max(first + (2 * HOUR), last) };
}

function percent(value, range) {
  return Math.max(0, Math.min(100, ((value - range.start) / (range.end - range.start)) * 100));
}

function timeLabel(seconds) {
  const wrapped = ((Math.round(seconds) % (24 * HOUR)) + (24 * HOUR)) % (24 * HOUR);
  const hours = Math.floor(wrapped / HOUR);
  const minutes = Math.floor((wrapped % HOUR) / MINUTE);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
}

function operationalStatus(today, locale) {
  const attention = Number(today?.health?.attention_count || 0);
  const targetState = today?.target?.state;
  if (attention > 1) return pageText(locale, `${attention} items require review before the next plan run.`, `${attention} נושאים דורשים בדיקה לפני הרצת התכנון הבאה.`);
  if (attention === 1) return pageText(locale, 'One item requires review before the next plan run.', 'נושא אחד דורש בדיקה לפני הרצת התכנון הבאה.');
  if (targetState !== 'set') return pageText(locale, 'No revenue target is set for this plan window.', 'לא הוגדר יעד הכנסה לחלון התכנון הזה.');
  return pageText(locale, 'No measured blockers in the saved plan.', 'לא נמצאו חסמים מדודים בתוכנית השמורה.');
}

function planScope(today, locale) {
  const candidates = [today?.decisions?.scope, today?.money?.scope];
  const scope = candidates.find((item) => item?.date_from && item?.date_to);
  return scope ? formatSpan(scope.date_from, scope.date_to, locale) : '';
}

export function TransmissionRibbon({ today, schedule, locale, onOpenPlan }) {
  const programs = useMemo(() => displayPrograms(schedule), [schedule]);
  const range = useMemo(() => rangeFor(programs), [programs]);
  const breaks = Array.isArray(schedule?.break_operations?.breaks) ? schedule.break_operations.breaks : [];
  const [selectedId, setSelectedId] = useState(() => programs.find((program) => Number(program.break_markers || 0) > 0)?.id || programs[0]?.id || '');
  const selected = programs.find((program) => program.id === selectedId) || programs[0] || null;
  const selectedBreaks = selected ? breaks.filter((item) => item.program_id === selected.id) : [];
  const visibleBreaks = range
    ? breaks.filter((item) => item.date === programs[0]?.date && Number(item.start_seconds) >= range.start && Number(item.start_seconds) <= range.end)
    : [];
  const money = today?.money || {};
  const attention = Number(today?.health?.attention_count || 0);
  const scope = planScope(today, locale);
  const timezone = today?.money?.scope?.timezone
    || today?.decisions?.scope?.timezone
    || schedule?.timezone
    || today?.timezone
    || '';
  const planRunAt = today?.plan_run_at ? formatDay(String(today.plan_run_at).slice(0, 10)) : '';
  const ticks = range
    ? Array.from({ length: Math.floor((range.end - range.start) / HOUR) + 1 }, (_, index) => range.start + (index * HOUR))
    : [];

  return (
    <section className="control-room-hero" aria-labelledby="control-room-title">
      <div className="control-room-summary">
        <div className="control-room-copy">
          <span className="control-room-kicker"><Radio size={15} /> {pageText(locale, 'Operational status · plan of record', 'תמונת מצב תפעולית · תוכנית בתוקף')}</span>
          <h1 id="control-room-title">
            {pageText(locale, 'Broadcast status', 'מצב השידור')}
            {(programs[0]?.channel || today?.channel) ? <> · <Name>{programs[0]?.channel || today?.channel}</Name></> : null}
          </h1>
          <p>{operationalStatus(today, locale)}</p>
        </div>
        <dl className="control-room-record" aria-label={pageText(locale, 'Plan of record scope', 'היקף התוכנית שבתוקף')}>
          <div>
            <dt>{pageText(locale, 'Plan window', 'חלון התוכנית')}</dt>
            <dd><Numeric>{scope || pageText(locale, 'Not recorded', 'לא תועד')}</Numeric></dd>
          </div>
          <div>
            <dt>{pageText(locale, 'Last plan run', 'הרצת תוכנית אחרונה')}</dt>
            <dd><Numeric>{planRunAt || pageText(locale, 'Not recorded', 'לא תועדה')}</Numeric></dd>
          </div>
        </dl>
        <div className="control-room-metric">
          <span>{pageText(locale, 'Saved-plan revenue', 'הכנסה לפי התוכנית')}</span>
          <strong><Numeric>{formatCurrency(money.amount_ils, locale)}</Numeric></strong>
          <small>{attention ? pageText(locale, `${formatNumber(attention, locale)} items need attention`, `${formatNumber(attention, locale)} נושאים דורשים טיפול`) : pageText(locale, 'No measured blockers', 'אין חסמים שנמדדו')}</small>
        </div>
      </div>

      <div className="card transmission-instrument">
        <div className="transmission-head">
          <div>
            <span>{pageText(locale, 'Broadcast timeline', 'ציר שידור')}</span>
            <strong>{programs[0]?.channel || today?.channel || pageText(locale, 'Saved plan', 'תוכנית שמורה')}</strong>
          </div>
          <div className="transmission-head-meta">
            <span>{pageText(locale, `${formatNumber(programs.length, locale)} programmes · ${formatNumber(visibleBreaks.length, locale)} breaks`, `${formatNumber(programs.length, locale)} תוכניות · ${formatNumber(visibleBreaks.length, locale)} ברייקים`)}</span>
            {programs[0]?.date ? <time dateTime={programs[0].date}>{formatDay(programs[0].date)}</time> : null}
          </div>
        </div>

        {range ? (
          <>
            <p className="transmission-clock-context" id="transmission-clock-context">
              {pageText(locale, 'Programme times · 24-hour plan clock', 'שעות התוכניות · שעון תוכנית של 24 שעות')}
              {' · '}
              {timezone
                ? <Code>{timezone}</Code>
                : pageText(locale, 'time zone not recorded', 'אזור הזמן לא תועד')}
            </p>
            <div className="transmission-stage chart-ltr" aria-describedby="transmission-clock-context">
              <div className="transmission-scale" aria-hidden="true">
                {ticks.map((tick) => (
                  <span key={tick} style={{ '--tick': `${percent(tick, range)}%` }}>{timeLabel(tick)}</span>
                ))}
              </div>
              <div className="transmission-track" aria-label={pageText(locale, 'Programmes in chronological order', 'תוכניות לפי סדר כרונולוגי')}>
                {programs.map((program) => {
                  const active = selected?.id === program.id;
                  const start = percent(Number(program.start_seconds), range);
                  const end = percent(Number(program.end_seconds), range);
                  return (
                    <Pressable
                      key={program.id}
                      type="button"
                      className={`transmission-program${active ? ' active' : ''}`}
                      style={{ '--start': `${start}%`, '--span': `${Math.max(2, end - start)}%` }}
                      aria-pressed={active}
                      onClick={() => setSelectedId(program.id)}
                    >
                      <time>{program.start_time}</time>
                      <Name>{program.title}</Name>
                    </Pressable>
                  );
                })}
                {visibleBreaks.map((item) => (
                  <span
                    key={item.id}
                    className="transmission-break"
                    style={{ '--break': `${percent(Number(item.start_seconds), range)}%` }}
                    title={`${item.start_time} · ${item.program_title}`}
                    aria-hidden="true"
                  />
                ))}
                {selected ? (
                  <span className="transmission-cursor" style={{ '--cursor': `${percent(Number(selected.start_seconds), range)}%` }} aria-hidden="true" />
                ) : null}
              </div>
            </div>
          </>
        ) : (
          <div className="transmission-empty" role="status">
            <CircleAlert size={18} />
            {pageText(locale, 'The saved plan has no timed programme trace to show.', 'בתוכנית השמורה אין עקבת תוכניות מתוזמנת להצגה.')}
          </div>
        )}

        {selected ? (
          <div className="transmission-readout" aria-live="polite">
            <div>
              <span>{selected.start_time}–{selected.end_time}</span>
              <strong><Name>{selected.title}</Name></strong>
            </div>
            <dl>
              <div><dt>{pageText(locale, 'Revenue', 'הכנסה')}</dt><dd><Numeric>{formatCurrency(selected.revenue, locale)}</Numeric></dd></div>
              <div><dt>{pageText(locale, 'Retention', 'שימור')}</dt><dd><Numeric>{formatPercent(selected.retention, locale)}</Numeric></dd></div>
              <div><dt>{pageText(locale, 'Breaks', 'ברייקים')}</dt><dd><Numeric>{formatNumber(selectedBreaks.length, locale)}</Numeric></dd></div>
            </dl>
            <Button variant="text" className="transmission-open" onClick={onOpenPlan} endIcon={<ArrowUpRight size={16} />}>
              {pageText(locale, 'Open in Plan', 'פתיחה בתכנון')}
            </Button>
          </div>
        ) : null}
      </div>
    </section>
  );
}

export default TransmissionRibbon;
