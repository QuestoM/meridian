import React from 'react';
import { Button } from '@mui/material';
import { Numeric, formatCurrency, formatPercent, pageText } from '../../shell/format';
import { dayLabel, daypartLabel } from '../../shell/labels';
import { daypartForTime, daypartKeys, flattenScheduleRows } from '../../shell/plan-model';

export function DaypartView({ rows, locale, selectedProgramKey, onSelectProgram }) {
  const programs = flattenScheduleRows(rows);
  const groups = daypartKeys.map((daypart) => ({
    daypart,
    items: programs.filter((program) => daypartForTime(program.time) === daypart),
  }));
  const populatedGroups = groups.filter((group) => group.items.length > 0);
  const emptyGroups = groups.filter((group) => group.items.length === 0);
  return (
    <div className="daypart-view">
      {populatedGroups.map(({ daypart, items }) => {
        const revenue = items.reduce((sum, program) => sum + Number(program.revenue || 0), 0);
        const avgRetention = items.length
          ? items.reduce((sum, program) => sum + Number(program.retention || 0), 0) / items.length
          : 0;
        return (
          <section className="daypart-card" key={daypart}>
            <div className="daypart-card-head">
              <div>
                <strong>{daypartLabel(daypart, locale)}</strong>
                <span>{items.length} {pageText(locale, 'programs', 'תוכניות')}</span>
              </div>
              <div>
                <strong><Numeric>{formatCurrency(revenue, locale)}</Numeric></strong>
                <span><Numeric>{formatPercent(avgRetention, locale)}</Numeric></span>
              </div>
            </div>
            <div className="daypart-programs">
              {items.slice(0, 7).map((program) => (
                <Button
                  key={program.key}
                  className={program.key === selectedProgramKey ? 'daypart-program active' : 'daypart-program'}
                  type="button"
                  variant="text"
                  onClick={() => onSelectProgram(program)}
                >
                  <span>{program.title}</span>
                  <small>{program.channel} / {dayLabel(program.day, locale)} / <Numeric>{program.time}</Numeric></small>
                  <strong><Numeric>{formatCurrency(program.revenue, locale)}</Numeric></strong>
                </Button>
              ))}
            </div>
          </section>
        );
      })}
      {emptyGroups.length > 0 && (
        <section className="daypart-empty-summary">
          <strong>{pageText(locale, 'No planned inventory', 'אין מלאי מתוכנן')}</strong>
          <span>
            {emptyGroups.map((group) => daypartLabel(group.daypart, locale)).join(' / ')}
          </span>
        </section>
      )}
    </div>
  );
}

export default DaypartView;
