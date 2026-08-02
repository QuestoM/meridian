import React from 'react';
import { Tooltip } from '@mui/material';
import { ChevronLeft, ChevronRight, Info, Layers, TriangleAlert } from 'lucide-react';
import { pageText } from './advertisers-helpers';
import { displayNameOf, isUnnamed, showsRawIdLine } from './advertiser-name-helpers';
import {
  EFFECT_META,
  conflictCount,
  effectMeta,
  formatPremium,
  premiumDelta,
  revenuePendingTooltip,
  revenueProvenance,
  totalRules,
} from './advertiser-stats-helpers';
import { exactMoney } from './clients-money-helpers';

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
      <Tooltip title={pageText(locale, 'The rule breakdown is still loading', 'פירוט הכללים עדיין נטען')} arrow placement="bottom">
        <span className="amz-effect-pending">
          {pageText(locale, 'breakdown pending', 'פירוט בטעינה')}
        </span>
      </Tooltip>
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
  const shownName = displayNameOf(row, locale);
  const unnamed = isUnnamed(row);
  const showRawId = showsRawIdLine(row, locale);

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
        `Open ${shownName} (${row.advertiser_id}) management area`,
        `פתיחת אזור הניהול של ${shownName} (${row.advertiser_id})`,
      )}
    >
      <header className="amz-card-head">
        <div className="amz-card-id-wrap">
          <span className={`amz-card-name${unnamed ? ' unnamed' : ''}`} dir="auto">
            {shownName}
            {unnamed && (
              <Tooltip title={pageText(locale, 'This pricing row carries no advertiser name, so it prices nobody. A client gets its rule from its own record, under Clients.', 'שורת התמחור הזו אינה נושאת שם מפרסם, ולכן היא אינה מתמחרת אף אחד. לקוח מקבל את הכלל שלו מהכרטיס שלו, במסך הלקוחות.')} arrow placement="top">
                <span className="amz-unnamed-chip">{pageText(locale, 'prices nobody', 'לא מתמחר אף אחד')}</span>
              </Tooltip>
            )}
          </span>
          {/* The raw id stays visible as a quiet secondary line whenever it differs from the shown name. */}
          {showRawId && <span className="amz-card-rawid" dir="ltr">{row.advertiser_id}</span>}
          {/* Native title is a truncation echo of the ellipsised notes, not an explanation. */}
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
            {conflicts === 1
              ? pageText(locale, '1 conflict', 'התנגשות אחת')
              : pageText(locale, `${conflicts} conflicts`, `${conflicts} התנגשויות`)}
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
          value={row.revenue === null || row.revenue === undefined ? null : exactMoney(row.revenue, locale)}
          provenance={revenueProvenance(row, locale)}
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
