import React from 'react';
import { Tooltip } from '@mui/material';
import { ChevronLeft, ChevronRight, Info, Layers, TriangleAlert } from 'lucide-react';
import { pageText } from './advertisers-helpers';
import {
  EFFECT_META,
  conflictCount,
  effectMeta,
  formatPremium,
  premiumDelta,
  revenuePendingTooltip,
  totalRules,
} from './advertiser-stats-helpers';

// One stat with an Info affordance carrying provenance. Value renders dir=ltr so
// numbers stay readable in the RTL Hebrew layout. A missing value shows "-".
function StatBlock({ label, value, provenance, tone, delta }) {
  const shown = value === null || value === undefined || value === '' ? '-' : value;
  const isEmpty = shown === '-';
  return (
    <div className="amz-stat-block">
      <span className="amz-stat-label">
        {label}
        <Tooltip title={provenance} arrow placement="top">
          <span className="amz-stat-info" tabIndex={0} role="img" aria-label={provenance}>
            <Info size={11} />
          </span>
        </Tooltip>
      </span>
      <span className={`amz-stat-figure ${tone || ''}${isEmpty ? ' empty' : ''}`} dir="ltr">
        {shown}
        {delta && <span className="amz-stat-delta">{delta}</span>}
      </span>
    </div>
  );
}

// A colour-coded chip per effect type that has at least one rule. The colour
// (teal / blue / red / muted) is sourced from EFFECT_META so the mix is legible
// at a glance. When no breakdown has loaded yet the chips are omitted entirely.
function EffectChips({ breakdown, locale }) {
  if (!breakdown) {
    return (
      <span className="amz-effect-pending" title={pageText(locale, 'Loading rule breakdown', 'טוען פירוט כללים')}>
        {pageText(locale, 'breakdown pending', 'פירוט בטעינה')}
      </span>
    );
  }
  const active = EFFECT_META.filter((meta) => Number(breakdown[meta.key] || 0) > 0);
  if (active.length === 0) {
    return <span className="amz-effect-none">{pageText(locale, 'no scoped rules', 'אין כללים ממוקדים')}</span>;
  }
  return (
    <div className="amz-effect-chips" role="list" aria-label={pageText(locale, 'Effect mix', 'תמהיל השפעות')}>
      {active.map((meta) => (
        <span key={meta.key} className={`amz-effect-chip ${meta.tone}`} role="listitem">
          <span className="amz-effect-name">{pageText(locale, meta.en, meta.he)}</span>
          <span className="amz-effect-count" dir="ltr">{breakdown[meta.key]}</span>
        </span>
      ))}
    </div>
  );
}

// A fully clickable advertiser management card. The whole surface opens the
// detail drawer; a clear rule-count cluster and effect mix make the state
// legible from the outside per the owner's ask. Revenue/profitability are
// pending and honestly shown as "-" with a provenance tooltip.
function AdvertiserStatCard({ row, locale, onOpen }) {
  const rules = totalRules(row);
  const conflicts = conflictCount(row);
  const baseline = row.baseline_premium ?? row.default_premium;
  const effective = row.avg_effective_premium;
  const Caret = locale === 'he' ? ChevronLeft : ChevronRight;

  const open = () => onOpen(row.advertiser_id);
  const onKeyDown = (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      open();
    }
  };

  return (
    <article
      className={`amz-card${conflicts > 0 ? ' has-conflict' : ''}`}
      role="button"
      tabIndex={0}
      onClick={open}
      onKeyDown={onKeyDown}
      aria-label={pageText(
        locale,
        `Open ${row.advertiser_id} management area`,
        `פתיחת אזור הניהול של ${row.advertiser_id}`,
      )}
    >
      <header className="amz-card-head">
        <div className="amz-card-id-wrap">
          <span className="amz-card-id" dir="ltr">{row.advertiser_id}</span>
          {row.notes ? <span className="amz-card-notes" title={row.notes}>{row.notes}</span> : null}
        </div>
        <Caret size={18} className="amz-card-caret" aria-hidden="true" />
      </header>

      <div className="amz-card-rulecount">
        <Layers size={16} className="amz-rulecount-icon" aria-hidden="true" />
        <span className="amz-rulecount-value" dir="ltr">{rules}</span>
        <span className="amz-rulecount-label">
          {rules === 1
            ? pageText(locale, 'scoped rule', 'כלל ממוקד')
            : pageText(locale, 'scoped rules', 'כללים ממוקדים')}
        </span>
        {conflicts > 0 && (
          <span className="amz-conflict-flag" dir="ltr">
            <TriangleAlert size={13} aria-hidden="true" />
            {pageText(locale, `${conflicts} conflict`, `${conflicts} התנגשות`)}
          </span>
        )}
      </div>

      <EffectChips breakdown={row.effect_breakdown} locale={locale} />

      <div className="amz-card-stats">
        <StatBlock
          label={pageText(locale, 'Baseline premium', 'מקדם בסיס')}
          value={formatPremium(baseline)}
          delta={premiumDelta(baseline)}
          tone={Number(baseline ?? 1) > 1 ? 'teal' : Number(baseline ?? 1) < 1 ? 'amber' : ''}
          provenance={pageText(
            locale,
            'Source: advertiser_rules.csv (the advertiser default premium)',
            'מקור: advertiser_rules.csv (מקדם ברירת המחדל של המפרסם)',
          )}
        />
        <StatBlock
          label={pageText(locale, 'Avg effective', 'מקדם אפקטיבי')}
          value={formatPremium(effective)}
          delta={premiumDelta(effective)}
          tone={Number(effective ?? 1) > 1 ? 'teal' : Number(effective ?? 1) < 1 ? 'amber' : ''}
          provenance={pageText(
            locale,
            'Source: rule engine - baseline times every ANY-scope premium rule. A real multiplier, not an estimate.',
            'מקור: מנוע הכללים - הבסיס כפול כל כלל מקדם בהיקף ״הכול״. מכפיל אמיתי, לא הערכה.',
          )}
        />
        <StatBlock
          label={pageText(locale, 'Revenue', 'הכנסה')}
          value={null}
          provenance={revenuePendingTooltip(locale)}
        />
        <StatBlock
          label={pageText(locale, 'Profitability', 'רווחיות')}
          value={null}
          provenance={revenuePendingTooltip(locale)}
        />
      </div>
    </article>
  );
}

export default AdvertiserStatCard;
