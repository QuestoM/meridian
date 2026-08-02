import React from 'react';
import { Numeric } from '../../shell/format';
import { Absent, Figure, Panel, Stat } from './console-bits';
import { pick, t } from './console-words';

// How much contrast the data carries, and the register of what is blocked on
// data nobody has. The register is the part that does not exist anywhere in the
// product today: per blocked factor, the condition that would end the block and
// the first date on which it could, computed from the checked-in calendar and
// the operator's own event store rather than estimated.

function Window({ window: block, locale }) {
  if (!block || !block.available) {
    return <Absent title={t('coverage.window', locale)} reason={pick(block, 'reason', locale)} />;
  }
  return (
    <Panel title={t('coverage.window', locale)}>
      <div className="mc-window-headline">{pick(block, 'headline', locale)}</div>
      <div className="mc-stat-row">
        <Stat
          label={t('coverage.window', locale)}
          value={<span dir="ltr"><Numeric>{`${block.start} .. ${block.end}`}</Numeric></span>}
          sub={`${block.days} ${t('coverage.days', locale)}`}
        />
        <Stat
          label={t('coverage.breaks', locale)}
          value={<Numeric>{Number(block.total_breaks_measured || 0).toLocaleString('en-US')}</Numeric>}
        />
        <Stat
          label={locale === 'en' ? 'After the ceasefire' : 'אחרי הפסקת האש'}
          value={<Numeric>{`${block.post_ceasefire_breaks} (${block.post_ceasefire_pct}%)`}</Numeric>}
          sub={<span dir="ltr">{block.ceasefire_date}</span>}
        />
      </div>
    </Panel>
  );
}

function RetentionContrast({ block, locale }) {
  if (!block || !block.available) {
    return <Absent title={t('coverage.retention', locale)} reason={pick(block, 'reason', locale)} />;
  }
  return (
    <Panel title={t('coverage.retention', locale)} sub={block.pooling_method}>
      <div className="mc-stat-row">
        <Stat label={t('coverage.cells', locale)} value={<Numeric>{String(block.cells)}</Numeric>} />
        <Stat
          label={t('coverage.per_cell', locale)}
          value={<Numeric>{`${block.per_cell_min} / ${block.per_cell_median} / ${block.per_cell_max}`}</Numeric>}
          sub={`${block.cells_under_ten} ${locale === 'en' ? 'cells under ten' : 'תאים מתחת לעשר'}`}
        />
        <Stat
          label={t('coverage.ratio', locale)}
          value={<Figure value={block.contrast_ratio} unit="ratio" digits={6} />}
          sub={<span dir="ltr"><Numeric>{`tau^2 ${block.between_cell_variance_tau2} / ${block.pooled_within_variance}`}</Numeric></span>}
        />
      </div>
      <p className="mc-note">{pick(block, 'note', locale)}</p>
    </Panel>
  );
}

function AudienceContrast({ block, locale }) {
  if (!block || !block.available) {
    return <Absent title={t('coverage.audience', locale)} reason={pick(block, 'reason', locale)} />;
  }
  return (
    <Panel title={t('coverage.audience', locale)} sub={block.kind}>
      <div className="mc-stat-row">
        <Stat
          label={locale === 'en' ? 'Observations' : 'תצפיות'}
          value={<Numeric>{Number(block.observations || 0).toLocaleString('en-US')}</Numeric>}
        />
        <Stat
          label={t('coverage.channels_counted', locale)}
          value={<Numeric>{String(block.channels_in_base)}</Numeric>}
          sub={block.operator_channel}
        />
        {(block.factor_levels || []).filter((row) => row.levels).map((row) => (
          <Stat
            key={row.factor}
            label={<span dir="ltr">{row.factor}</span>}
            value={<Numeric>{Number(row.levels).toLocaleString('en-US')}</Numeric>}
            sub={t('coverage.levels', locale)}
          />
        ))}
      </div>
      <p className="mc-note">{pick(block, 'note', locale)}</p>
    </Panel>
  );
}

function BlockedRegister({ rows, locale }) {
  if (!rows || rows.length === 0) {
    return (
      <Panel title={t('coverage.blocked', locale)}>
        <Absent
          title={locale === 'en' ? 'Nothing is blocked on missing data.' : 'שום דבר אינו חסום על נתונים חסרים.'}
        />
      </Panel>
    );
  }
  return (
    <Panel title={t('coverage.blocked', locale)} sub={`${rows.length}`}>
      <ul className="mc-blocked-list">
        {rows.map((row) => (
          <li className="mc-blocked" key={row.gate_id}>
            <div className="mc-blocked-head">
              <strong>{locale === 'en' ? row.label_en : row.label_he}</strong>
              {row.earliest_state === 'dated' ? (
                <span className="mc-blocked-date">
                  {t('coverage.earliest', locale)} <span dir="ltr"><Numeric>{row.earliest.start}</Numeric></span>
                </span>
              ) : (
                <span className="mc-blocked-date mc-unknown">{t('coverage.earliest_unknown', locale)}</span>
              )}
            </div>
            <p className="mc-blocked-condition">
              <span className="mc-blocked-label">{t('coverage.condition', locale)}</span>
              {pick(row, 'condition', locale)}
            </p>
            {Object.keys(row.evidence || {}).length ? (
              <p className="mc-blocked-evidence" dir="ltr">
                {Object.entries(row.evidence)
                  .filter(([, value]) => value !== null && value !== undefined)
                  .map(([key, value]) => `${key}: ${value}`)
                  .join('   ')}
              </p>
            ) : null}
            <p className="mc-blocked-source">
              {t('coverage.from', locale)} <code dir="ltr">{row.source}</code>
            </p>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

export default function CoveragePanel({ payload, locale }) {
  return (
    <>
      <Window window={payload.window} locale={locale} />
      <RetentionContrast block={payload.retention} locale={locale} />
      <AudienceContrast block={payload.audience} locale={locale} />
      <BlockedRegister rows={payload.blocked} locale={locale} />
    </>
  );
}
