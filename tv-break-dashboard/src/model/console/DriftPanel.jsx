import React, { useState } from 'react';
import { Numeric } from '../../shell/format';
import { Absent, Figure, Panel, RecordDrill, Stat } from './console-bits';
import { pick, t } from './console-words';

// What drifted. The weekly measurement is real and complete; the series across
// model versions is not, and the reason is stated rather than smoothed over: a
// training used to overwrite the artifact in place, so no earlier version was
// kept. The store now records one per version, so the series starts here.

function percent(value) {
  if (value === null || value === undefined) return null;
  return Number(value) * 100;
}

export default function DriftPanel({ payload, locale }) {
  const [open, setOpen] = useState(false);
  const current = payload.current || {};
  if (current.status !== 'measured') {
    return (
      <Panel title={t('section.drift', locale)}>
        <Absent
          title={locale === 'en'
            ? 'No level-drift measurement is available for the current model.'
            : 'אין מדידת סחיפת רמה עבור המודל הנוכחי.'}
          reason={current.reason || ''}
        />
      </Panel>
    );
  }
  const weeks = current.weekly_levels || [];
  const means = weeks.map((week) => Number(week.mean_log_effect)).filter((value) => Number.isFinite(value));
  const low = means.length ? Math.min(...means) : 0;
  const span = means.length ? Math.max(...means) - low : 0;
  const binding = current.binding === true;
  return (
    <>
      <Panel
        title={t('section.drift', locale)}
        sub={`${current.n_weeks} ${t('coverage.days', locale) === 'days' ? 'weeks' : 'שבועות'}, ${current.n_breaks} ${t('coverage.breaks', locale)}`}
        right={(
          <span className={`mc-verdict ${binding ? 'mc-tested_and_lost' : 'mc-active'} mc-md`}>
            {binding ? t('drift.binding', locale) : t('drift.stable', locale)}
          </span>
        )}
      >
        <div className="mc-stat-row">
          <Stat
            label={t('drift.per_week', locale)}
            value={<Figure value={percent(current.drift_per_week)} unit="percent" />}
            sub={<span dir="ltr"><Numeric>{`se ${(Number(current.drift_se) * 100).toFixed(2)}%`}</Numeric></span>}
          />
          <Stat
            label={t('drift.threshold', locale)}
            value={<Figure value={percent(current.binding_threshold)} unit="percent" />}
          />
          <Stat
            label={locale === 'en' ? 'Slope per week' : 'שיפוע לשבוע'}
            value={<Figure value={percent(current.slope_per_week)} unit="percent" />}
            sub={<span dir="ltr"><Numeric>{`se ${(Number(current.slope_se) * 100).toFixed(2)}%`}</Numeric></span>}
          />
        </div>
        <p className="mc-note">
          <span className="mc-basis-label">{t('drift.criterion', locale)}</span>
          <span dir="ltr">{current.criterion}</span>
        </p>
        <div className="mc-week-strip">
          <span className="mc-week-caption">{t('drift.weekly', locale)}</span>
          {weeks.map((week, index) => {
            const mean = Number(week.mean_log_effect);
            const ratio = span <= 0 || !Number.isFinite(mean) ? 1 : (mean - low) / span;
            return (
              <div className="mc-week" key={`week-${week.week ?? index}`}>
                <small>{t('drift.week', locale)} <Numeric>{String(week.week ?? index + 1)}</Numeric></small>
                <span className="mc-week-bar" aria-hidden="true">
                  <i style={{ '--mc-week-width': `${Math.round(12 + ratio * 88)}%` }} />
                </span>
                <strong><Figure value={percent(mean)} unit="percent" /></strong>
                <small><Numeric>{`n=${week.n}`}</Numeric></small>
              </div>
            );
          })}
        </div>
        <RecordDrill record={current} locale={locale} open={open} onToggle={() => setOpen((v) => !v)} />
      </Panel>
      <Panel title={t('drift.series', locale)} sub={`${(payload.series || []).length}`}>
        {(payload.series || []).length > 1 ? (
          <ul className="mc-series">
            {payload.series.map((point) => (
              <li key={point.model_version_id}>
                <span dir="ltr">{point.name}</span>
                <Figure value={percent(point.drift_per_week)} unit="percent" />
              </li>
            ))}
          </ul>
        ) : (
          <Absent
            title={locale === 'en'
              ? 'One point is not a series.'
              : 'נקודה אחת אינה סדרה.'}
            reason={pick(payload, 'series_reason', locale)}
          />
        )}
      </Panel>
    </>
  );
}
