import React from 'react';
import { Code } from '../../shell/bidi';
import { Numeric } from '../../shell/format';
import { Absent, Earliest, Figure, Panel, Stat } from './console-bits';
import { pick, t } from './console-words';
import { Pressable } from '../../studio/dom-controls';

// How much contrast the data carries, and the register of what is blocked on
// data nobody has. The register is the part that does not exist anywhere in the
// product today: per blocked factor, the condition that would end the block and
// the first date on which it could, computed from the checked-in calendar and
// the operator's own event store rather than estimated.
//
// The register is also the one screen on this surface that used to end in prose.
// Measured on the shipped console: every other section carried between one and
// twenty-one controls and this one carried none, while the payload behind it
// already named the thing that ends each block, the date range it runs over, and
// the store two of the five rows are waiting on. Section 3.6 makes each of those
// a control or a stated tri-state, which is what the three parts below do.

// Which store a blocked row's source names, and therefore who can end the block.
//
// The source is a real address, so it is classified rather than printed. The
// operator's own event store is a page in this product and gets the control that
// opens it; a file checked in with the product cannot be supplied by anybody, so
// the honest answer is that only time ends that block and the screen says so. A
// source this table does not know reads unknown rather than guessing, and a test
// asserts the live register carries no unknown, so a new source is a red test
// rather than a silent shrug on screen.
const SUPPLY = {
  'data/calendar_events.csv': 'store',
  'kairos/config/israel_calendar.csv': 'time',
  'kairos/data/israel_calendar.py season bands': 'time',
  'the training window itself': 'time',
};

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
          value={<Numeric>{`${block.start} .. ${block.end}`}</Numeric>}
          sub={`${block.days} ${t('coverage.days', locale)}`}
        />
        <Stat
          label={t('coverage.breaks', locale)}
          value={<Numeric>{Number(block.total_breaks_measured || 0).toLocaleString('en-US')}</Numeric>}
        />
        <Stat
          label={locale === 'en' ? 'After the ceasefire' : 'אחרי הפסקת האש'}
          value={<Numeric>{`${block.post_ceasefire_breaks} (${block.post_ceasefire_pct}%)`}</Numeric>}
          sub={<Numeric>{block.ceasefire_date}</Numeric>}
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
        {/*
          The two variances the ratio above is made of, at the precision the
          rest of the surface prints. They shipped raw: seventeen significant
          digits of float, on the one line of this panel that did not go through
          the console's own number bit, beside a ratio that was rounded.
        */}
        <Stat
          label={t('coverage.ratio', locale)}
          value={<Figure value={block.contrast_ratio} unit="ratio" digits={6} />}
          sub={(
            <span className="mc-variances">
              <Numeric>tau^2</Numeric>
              {' '}
              <Figure value={block.between_cell_variance_tau2} unit="ratio" digits={6} />
              {' '}
              <Numeric>/</Numeric>
              {' '}
              <Figure value={block.pooled_within_variance} unit="ratio" digits={6} />
              {' '}
              <span>{t('coverage.between_within', locale)}</span>
            </span>
          )}
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
            label={<Code>{row.factor}</Code>}
            value={<Numeric>{Number(row.levels).toLocaleString('en-US')}</Numeric>}
            sub={t('coverage.levels', locale)}
          />
        ))}
      </div>
      <p className="mc-note">{pick(block, 'note', locale)}</p>
    </Panel>
  );
}

// What was counted to reach the verdict on this row, in words.
//
// The register shipped these as the raw keys the payload carries them under, so
// the screen read "days_in_window: 30 event_free_days_in_window: 0". A key with
// an underscore in it is the inside of the program on the outside of it. A key
// this file has no word for keeps its raw name rather than being dropped,
// because a raw key is a defect somebody fixes and a missing count is one
// nobody can see.
function Counted({ evidence, locale }) {
  const rows = Object.entries(evidence || {})
    .filter(([, value]) => value !== null && value !== undefined);
  if (rows.length === 0) return null;
  return (
    <p className="mc-blocked-evidence">
      <span className="mc-blocked-label">{t('coverage.counted', locale)}</span>
      {' '}
      {/*
        The word first and the figure after it. Hebrew agrees in number, so a
        count rendered figure-first reads "1 עונות" whenever the count is one,
        which the register does carry. Read as a label and its value, both
        languages agree at every count.
      */}
      {rows.map(([key, value], index) => (
        <React.Fragment key={key}>
          {index ? ' ' : null}
          <span className="mc-blocked-count">
            {t(`coverage.evidence.${key}`, locale) || <code><Code>{key}</Code></code>}
            {' '}
            <Numeric>{Number.isFinite(Number(value)) ? Number(value).toLocaleString('en-US') : String(value)}</Numeric>
          </span>
        </React.Fragment>
      ))}
    </p>
  );
}

// The source, as the thing it is: an address somebody can open, or a file that
// arrives with the product, which nobody supplies and only more history ends.
// The path stays on screen either way, because it is what a steward checks the
// console against. What changes is whether it is the control that opens it.
function Supply({ row, locale, onOpenEvents }) {
  const supply = SUPPLY[row.source] || 'unknown';
  const openable = supply === 'store' && Boolean(onOpenEvents);
  return (
    <p className={`mc-blocked-source mc-supply-${supply}`}>
      <span className="mc-blocked-label">{t('coverage.supply', locale)}</span>
      {' '}
      {openable ? (
        <Pressable type="button" className="mc-link mc-blocked-open" onClick={onOpenEvents}>
          {t('coverage.supply_store_open', locale)}
          {' '}
          <code><Code>{row.source}</Code></code>
        </Pressable>
      ) : (
        <code><Code>{row.source}</Code></code>
      )}
      {' '}
      <span className="mc-blocked-supply">{t(`coverage.supply_${supply}`, locale)}</span>
    </p>
  );
}

function BlockedRegister({ rows, locale, onOpenEvents }) {
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
                  <Earliest earliest={row.earliest} locale={locale} />
                </span>
              ) : (
                <span className="mc-blocked-date mc-unknown">{t('coverage.earliest_unknown', locale)}</span>
              )}
            </div>
            <p className="mc-blocked-condition">
              <span className="mc-blocked-label">{t('coverage.condition', locale)}</span>
              {pick(row, 'condition', locale)}
            </p>
            <Counted evidence={row.evidence} locale={locale} />
            <Supply row={row} locale={locale} onOpenEvents={onOpenEvents} />
          </li>
        ))}
      </ul>
    </Panel>
  );
}

export default function CoveragePanel({ payload, locale, onOpenEvents }) {
  return (
    <>
      <Window window={payload.window} locale={locale} />
      <RetentionContrast block={payload.retention} locale={locale} />
      <AudienceContrast block={payload.audience} locale={locale} />
      <BlockedRegister rows={payload.blocked} locale={locale} onOpenEvents={onOpenEvents} />
    </>
  );
}
