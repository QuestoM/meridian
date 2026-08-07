import React, { useMemo } from 'react';
import { pageText } from '../shell/format';
import { Figure } from '../shell/bidi';
import HistoryRow from './HistoryRow';
import { dayHeading, isWeekend, isoDay } from './history-labels';

// The list. Entries arrive newest first and are grouped by broadcast day, with
// the day heading sticky while its rows scroll, which is the grouping Linear
// uses and the reason a long feed stays readable.
//
// The Israeli week is a law here, not a preference: the heading names the day
// from a Sunday-first table and Friday and Saturday carry the weekend mark, so
// a reader can see at a glance that a change landed on a weekend.

function groupByDay(entries) {
  const groups = [];
  const index = new Map();
  entries.forEach((entry) => {
    const day = isoDay(entry.ts);
    if (!index.has(day)) {
      index.set(day, groups.length);
      groups.push({ day, entries: [] });
    }
    groups[index.get(day)].entries.push(entry);
  });
  return groups;
}

export default function HistoryTimeline({ entries, locale, selectedId, onSelect, listRef }) {
  const groups = useMemo(() => groupByDay(entries), [entries]);
  let position = -1;

  return (
    <div className="hist-list" role="listbox" aria-label={pageText(locale, 'History', 'היסטוריה')} ref={listRef}>
      {groups.map((group) => {
        const date = new Date(`${group.day}T00:00:00`);
        const weekend = !Number.isNaN(date.getTime()) && isWeekend(date);
        return (
          <section className="hist-group" key={group.day}>
            <header className={`hist-day${weekend ? ' weekend' : ''}`}>
              <span className="hist-day-name">{dayHeading(group.day, locale)}</span>
              {weekend ? <span className="hist-day-mark">{pageText(locale, 'Weekend', 'סוף שבוע')}</span> : null}
              <span className="hist-day-count"><Figure>{group.entries.length}</Figure></span>
            </header>
            {group.entries.map((entry) => {
              position += 1;
              return (
                <HistoryRow
                  key={entry.id}
                  entry={entry}
                  index={position}
                  locale={locale}
                  selected={entry.id === selectedId}
                  onSelect={onSelect}
                />
              );
            })}
          </section>
        );
      })}
    </div>
  );
}
