import React from 'react';
import { Code, Name } from '../../shell/bidi';
import { Numeric } from '../../shell/format';
import { t } from './console-words';

// The console's small parts. Two of them carry the whole argument of the
// surface, so they live here rather than inside one panel:
//
// - `Verdict` renders a state as a state, never as a chip that means three
//   things. Colour, shape and word all move together, and the meaning sits
//   beside it in words rather than in a tooltip nobody opens.
// - `Basis` prints what a verdict was decided on, on the same row as the
//   verdict. Stripe's rule: the figure carries its basis, or the figure does
//   not render.

export const STATE_ORDER = ['active', 'tested_and_lost', 'no_contrast', 'not_measured'];

export function Verdict({ state, labelEn, labelHe, locale, size = 'md' }) {
  const label = locale === 'en' ? labelEn : labelHe;
  return <span className={`mc-verdict mc-${state} mc-${size}`}>{label}</span>;
}

// A number with its unit, isolated so a Latin run inside Hebrew text does not
// reorder. Renders the honest dash when there is no number rather than a zero.
export function Figure({ value, unit, digits = 2 }) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return <span className="mc-figure mc-absent">-</span>;
  }
  const number = Number(value);
  const text = unit === 'percent'
    ? `${number > 0 ? '+' : ''}${number.toFixed(digits)}%`
    : unit === 'p'
      ? `p=${number.toFixed(4)}`
      : number.toLocaleString('en-US', { maximumFractionDigits: digits });
  return <span className="mc-figure"><Numeric>{text}</Numeric></span>;
}

export function Money({ value, locale, digits = 0 }) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return <span className="mc-figure mc-absent">-</span>;
  }
  const number = Number(value);
  const sign = number > 0 ? '+' : '';
  const text = `${sign}${number.toLocaleString(locale === 'en' ? 'en-US' : 'he-IL', {
    maximumFractionDigits: digits,
  })} ₪`;
  return <span className={`mc-money ${number > 0 ? 'up' : number < 0 ? 'down' : 'flat'}`}><Numeric>{text}</Numeric></span>;
}

// The line under a verdict. Everything on it is measured; where the artifact
// carries no number the phrase says so instead of printing one.
export function Basis({ basis, locale }) {
  if (!basis) return null;
  const statistic = locale === 'en' ? basis.statistic_en : basis.statistic_he;
  const parts = [];
  if (basis.value === null || basis.value === undefined) {
    parts.push(t('gates.no_figure', locale));
  } else {
    parts.push(
      <span key="measured">
        {t('gates.measured', locale)} <Figure value={basis.value} unit={basis.unit} />
      </span>,
    );
  }
  if (basis.bar !== null && basis.bar !== undefined) {
    parts.push(
      <span key="bar">
        {t('gates.bar', locale)} <Figure value={basis.bar} unit={basis.bar_unit} digits={basis.bar_unit === 'p' ? 4 : 2} />
      </span>,
    );
  }
  if (basis.n) {
    parts.push(
      <span key="n">
        <Numeric>{Number(basis.n).toLocaleString('en-US')}</Numeric> {t('gates.observations', locale)}
      </span>,
    );
  }
  if (basis.folds) {
    parts.push(
      <span key="folds">
        <Numeric>{String(basis.folds)}</Numeric> {t('gates.folds', locale)}
      </span>,
    );
  }
  if (basis.measured_at) {
    parts.push(
      <span key="measured_at">
        {t('candidates.measured_at', locale)} <Numeric>{String(basis.measured_at).slice(0, 19)}</Numeric>
      </span>,
    );
  }
  return (
    <div className="mc-basis">
      <span className="mc-basis-label">{t('gates.basis', locale)}</span>
      <span className="mc-basis-statistic">{statistic}</span>
      <span className="mc-basis-parts">{parts}</span>
      {basis.bar_source ? (
        <span className="mc-basis-source">
          {t('gates.bar_from', locale)} <code><Code>{basis.bar_source}</Code></code>
        </span>
      ) : null}
    </div>
  );
}

// A raw artifact record, shown on demand. This is the second level of the
// drill: a verdict opens its basis, the basis opens the record it was read
// from, so no figure on this surface is unreachable from its source.
export function RecordDrill({ record, locale, open, onToggle, label }) {
  if (record === null || record === undefined) return null;
  return (
    <div className="mc-drill">
      <button type="button" className="mc-link" onClick={onToggle} aria-expanded={open}>
        {open ? t('gates.hide_record', locale) : (label || t('gates.show_record', locale))}
      </button>
      {open ? (
        <pre className="mc-record"><Code>{JSON.stringify(record, null, 1)}</Code></pre>
      ) : null}
    </div>
  );
}

// The first date a block could end, with the name of the thing that ends it.
//
// A bare date is a fact nobody can act on. Measured on the shipped console: the
// register printed 2024-12-26 alone on two rows while the payload it was reading
// carried the name of the thing arriving on that date and the date it runs to,
// so a screen that knew the block ends with חנוכה, on the eight days to
// 2025-01-02, showed the reader neither. Both are rendered here, once, so the
// gate table and the register cannot say different things about the same row.
//
// The name takes its direction from its own first strong character rather than
// from the field it arrives in. It is called name_he because the calendar is
// Hebrew, but the row it sits in is Latin whenever the steward reads in English,
// and a name stated as rtl there would drag the date beside it out of order.
export function Earliest({ earliest, locale }) {
  if (!earliest || !earliest.start) return null;
  const span = !earliest.end || earliest.end === earliest.start
    ? earliest.start
    : `${earliest.start} .. ${earliest.end}`;
  return (
    <span className="mc-earliest">
      <span className="mc-earliest-label">{t('coverage.earliest', locale)}</span>
      {' '}
      {earliest.name_he ? <Name className="mc-earliest-name">{earliest.name_he}</Name> : null}
      {' '}
      <span className="mc-earliest-span"><Numeric>{span}</Numeric></span>
    </span>
  );
}

export function Stat({ label, value, sub }) {
  return (
    <div className="mc-stat">
      <span className="mc-stat-label">{label}</span>
      <strong className="mc-stat-value">{value}</strong>
      {sub ? <small className="mc-stat-sub">{sub}</small> : null}
    </div>
  );
}

// An absence that names what is missing and what would supply it. Never a zero,
// never a placeholder figure.
export function Absent({ title, reason, action }) {
  return (
    <div className="mc-absent-state">
      <strong>{title}</strong>
      {reason ? <p>{reason}</p> : null}
      {action || null}
    </div>
  );
}

export function Panel({ title, sub, right, children }) {
  return (
    <section className="card mc-panel">
      <header className="mc-panel-head">
        <div>
          <h2>{title}</h2>
          {sub ? <p>{sub}</p> : null}
        </div>
        {right || null}
      </header>
      {children}
    </section>
  );
}
