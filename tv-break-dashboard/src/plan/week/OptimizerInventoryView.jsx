import React from 'react';
import { Button } from '../../studio/actions';
import { Numeric, formatCurrency, formatNumber, formatPercent, pageText } from '../../shell/format';
import { dayLabel } from '../../shell/labels';
import { normalizeRows, programKey } from '../../shell/plan-model';

export function OptimizerInventoryView({ rows, locale, selectedProgramKey, onSelectProgram }) {
  const channelRows = normalizeRows(rows).map((row) => {
    const programs = normalizeRows(row.programs).map((program) => ({
      ...program,
      channel: row.channel,
      key: programKey(row.channel, program),
    }));
    const revenue = programs.reduce((sum, program) => sum + Number(program.revenue || 0), 0);
    const breaks = programs.reduce((sum, program) => sum + Number(program.break_markers || 0), 0);
    const retention = programs.length
      ? programs.reduce((sum, program) => sum + Number(program.retention || 0), 0) / programs.length
      : 0;
    return { channel: row.channel, programs, revenue, breaks, retention };
  });
  const maxRevenue = Math.max(...channelRows.map((row) => row.revenue), 1);

  return (
    <div className="optimizer-inventory-view">
      {channelRows.map((row) => (
        <section className="inventory-channel-card" key={row.channel}>
          <div className="inventory-channel-head">
            <div>
              <strong>{row.channel}</strong>
              <span>{row.programs.length} {pageText(locale, 'programs', 'תוכניות')} / {formatNumber(row.breaks, locale)} {pageText(locale, 'breaks', 'ברייקים')}</span>
            </div>
            <strong><Numeric>{formatCurrency(row.revenue, locale)}</Numeric></strong>
          </div>
          <i className="inventory-pressure" style={{ '--bar': row.revenue / maxRevenue }} />
          <div className="inventory-channel-meta">
            <span>{pageText(locale, 'Avg retention', 'שימור ממוצע')}</span>
            <strong><Numeric>{formatPercent(row.retention, locale)}</Numeric></strong>
          </div>
          <div className="inventory-program-list">
            {row.programs.slice(0, 4).map((program) => (
              <Button
                key={program.key}
                className={program.key === selectedProgramKey ? 'inventory-program active' : 'inventory-program'}
                type="button"
                variant="text"
                onClick={() => onSelectProgram(program)}
              >
                <span>{program.title}</span>
                <small>{dayLabel(program.day, locale)} / <Numeric>{program.time}</Numeric></small>
              </Button>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

export default OptimizerInventoryView;
