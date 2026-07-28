import React, { useState } from 'react';
import { Button } from '@mui/material';
import { Layers, Lock, RefreshCcw, RotateCcw, SlidersHorizontal, Tag, TriangleAlert, Users } from 'lucide-react';
import { pageText, formatCurrency, formatNumber, finiteNumber } from './surface-helpers';

// One proposal batch from the assistant, rendered for explicit operator
// approval. The card never applies anything by itself: selection, an inline
// confirm step that names the automatic restore point, and the apply and
// reject calls all go through the parent, so the inline chat copy and the
// pending tab stay in sync. Missing fields render as honest fallbacks.

const KINDS = {
  settings_change: { Icon: SlidersHorizontal, en: 'Settings change', he: 'שינוי הגדרות' },
  constraint: { Icon: Lock, en: 'New constraint', he: 'אילוץ חדש' },
  override: { Icon: Layers, en: 'Override', he: 'עקיפה' },
  pricing_change: { Icon: Tag, en: 'Pricing change', he: 'שינוי תמחור' },
  advertiser_change: { Icon: Users, en: 'Advertiser change', he: 'שינוי מפרסם' },
  recompute: { Icon: RefreshCcw, en: 'Recompute', he: 'חישוב מחדש' },
};

const STATUS_LABELS = {
  pending: ['Pending', 'ממתין'],
  applied: ['Applied', 'הוחל'],
  failed: ['Failed', 'נכשל'],
  rejected: ['Rejected', 'נדחה'],
};

function shortValue(value) {
  if (value === null || value === undefined || value === '') return '-';
  if (typeof value === 'object') {
    const text = JSON.stringify(value);
    return text.length > 80 ? `${text.slice(0, 77)}…` : text;
  }
  return String(value);
}

// settings_change payloads become field rows. A {from,to} or {before,after}
// value renders as before to after; anything else shows the target value
// alone, because inventing a "current" value we never fetched would be a lie.
function settingsRows(payload) {
  if (!payload || typeof payload !== 'object') return [];
  if (Array.isArray(payload.changes)) {
    return payload.changes
      .filter((change) => change && typeof change === 'object')
      .map((change, index) => ({
        field: String(change.field ?? change.key ?? `#${index + 1}`),
        from: change.from ?? change.before ?? null,
        to: change.to ?? change.after ?? change.value ?? null,
      }));
  }
  const source = payload.changes && typeof payload.changes === 'object' ? payload.changes : payload;
  return Object.entries(source).map(([field, value]) => {
    if (value && typeof value === 'object' && !Array.isArray(value) && ('to' in value || 'after' in value)) {
      return { field, from: 'from' in value ? value.from : 'before' in value ? value.before : null, to: 'to' in value ? value.to : value.after };
    }
    return { field, from: null, to: value };
  });
}

function PayloadView({ item, locale }) {
  if (!item.payload) return null;
  if (item.kind === 'settings_change') {
    const rows = settingsRows(item.payload);
    if (!rows.length) return null;
    const shown = rows.slice(0, 8);
    return (
      <div className="asst-fields">
        {shown.map((row) => (
          <div className="asst-field-row" key={row.field}>
            <span className="asst-field-name" dir="ltr">{row.field}</span>
            <span className="asst-field-value" dir="ltr">
              {row.from !== null && row.from !== undefined ? `${shortValue(row.from)} → ${shortValue(row.to)}` : shortValue(row.to)}
            </span>
          </div>
        ))}
        {rows.length > shown.length ? (
          <span className="asst-field-more">{pageText(locale, `And ${rows.length - shown.length} more fields`, `ועוד ${rows.length - shown.length} שדות`)}</span>
        ) : null}
      </div>
    );
  }
  if (Array.isArray(item.payload)) return null;
  const entries = Object.entries(item.payload).slice(0, 6);
  if (!entries.length) return null;
  return (
    <div className="asst-fields">
      {entries.map(([key, value]) => (
        <div className="asst-field-row" key={key}>
          <span className="asst-field-name" dir="ltr">{key}</span>
          <span className="asst-field-value" dir="ltr">{shortValue(value)}</span>
        </div>
      ))}
    </div>
  );
}

// The clear before-and-after view of exactly which variables will change versus
// what was there. Any item may carry item.diff = [{field, before, after}]; it
// renders as a field | before | after table with ltr values. A null before means
// a fresh value, so the whole card carries a "new" badge (a creation, for example
// a brand-new advertiser), and each such before cell shows the empty sentinel
// rather than a fabricated prior value. For an advertiser change the header names
// the advertiser so the operator sees whose record is being touched.
function advertiserName(item) {
  const payload = item && item.payload && typeof item.payload === 'object' ? item.payload : {};
  const name = payload.advertiser_name ?? payload.advertiser ?? payload.name;
  return name ? String(name) : '';
}

function DiffView({ item, locale }) {
  const rows = Array.isArray(item.diff) ? item.diff.filter((row) => row && typeof row === 'object') : [];
  if (!rows.length) return null;
  const isCreation = rows.every((row) => row.before === null || row.before === undefined);
  const advName = item.kind === 'advertiser_change' ? advertiserName(item) : '';
  const header = advName || pageText(locale, 'Changes to apply', 'השינויים שיוחלו');
  return (
    <div className="asst-pdiff">
      <div className="asst-pdiff-head">
        <span dir="auto">{header}</span>
        {isCreation ? <span className="asst-pdiff-new">{pageText(locale, 'New', 'חדש')}</span> : null}
      </div>
      <div className="asst-pdiff-grid">
        <div className="asst-pdiff-row head">
          <span dir="auto">{pageText(locale, 'Field', 'שדה')}</span>
          <span dir="auto">{pageText(locale, 'Before', 'לפני')}</span>
          <span dir="auto">{pageText(locale, 'After', 'אחרי')}</span>
        </div>
        {rows.map((row, index) => (
          <div className="asst-pdiff-row" key={index}>
            <span className="asst-pdiff-field" dir="ltr">{String(row.field ?? '')}</span>
            <span className="asst-pdiff-before" dir="ltr">{row.before === null || row.before === undefined ? '-' : shortValue(row.before)}</span>
            <span className="asst-pdiff-after" dir="ltr">{shortValue(row.after)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// A settings_change item may carry a measured before/after effect on the owned
// channel, so the operator sees what a change would do before approving it. The
// three money lines use the same vocabulary as the rest of the product (gross
// revenue, retention cost, net) plus the breaks change, each with a signed
// delta. A missing figure is dropped rather than invented; an unavailable
// effect shows a quiet reason instead.
const EFFECT_METRICS = [
  ['gross', 'Gross revenue', 'הכנסות ברוטו'],
  ['retention_cost', 'Retention cost', 'עלות שימור'],
  ['net', 'Net', 'נטו'],
];

function signedFigure(value, locale, money) {
  const body = money ? formatCurrency(Math.abs(value), locale) : formatNumber(Math.abs(value), locale);
  return `${value > 0 ? '+' : value < 0 ? '-' : ''}${body}`;
}

function metricCells(beforeVal, afterVal, deltaVal, locale, money) {
  const fmt = (value) => (money ? formatCurrency(value, locale) : formatNumber(value, locale));
  const before = finiteNumber(beforeVal);
  const after = finiteNumber(afterVal);
  let delta = finiteNumber(deltaVal);
  if (delta === null && before !== null && after !== null) delta = after - before;
  const flow = before !== null && after !== null ? `${fmt(before)} → ${fmt(after)}` : after !== null ? fmt(after) : before !== null ? fmt(before) : '';
  return { shown: before !== null || after !== null || delta !== null, flow, delta: delta !== null ? signedFigure(delta, locale, money) : '' };
}

function EffectView({ effect, locale }) {
  if (!effect || typeof effect !== 'object') return null;
  const header = pageText(locale, 'What this change would do', 'מה השינוי הזה יעשה');
  if (effect.status === 'unavailable') {
    const reason = effect.reason ? String(effect.reason) : pageText(locale, 'A preview is not available for this change.', 'אין תצוגה מקדימה לשינוי הזה.');
    return <div className="asst-effect"><span className="asst-effect-head">{header}</span><p className="asst-effect-note" dir="auto">{reason}</p></div>;
  }
  const before = effect.before && typeof effect.before === 'object' ? effect.before : {};
  const after = effect.after && typeof effect.after === 'object' ? effect.after : {};
  const delta = effect.delta && typeof effect.delta === 'object' ? effect.delta : {};
  const rows = EFFECT_METRICS.map(([key, en, he]) => ({ key, label: pageText(locale, en, he), ...metricCells(before[key], after[key], delta[key], locale, true) })).filter((row) => row.shown);
  const breaks = metricCells(before.breaks, after.breaks, delta.breaks, locale, false);
  if (!rows.length && !breaks.shown) return null;
  return (
    <div className="asst-effect">
      <span className="asst-effect-head">{header}</span>
      {rows.map((row) => (
        <div className={`asst-effect-row${row.key === 'net' ? ' net' : ''}`} key={row.key}>
          <span className="asst-effect-label" dir="auto">{row.label}</span>
          <span className="asst-effect-flow" dir="ltr">{row.flow || '-'}</span>
          <span className="asst-effect-delta" dir="ltr">{row.delta}</span>
        </div>
      ))}
      {breaks.shown ? (
        <div className="asst-effect-row" key="breaks">
          <span className="asst-effect-label" dir="auto">{pageText(locale, 'Breaks', 'ברייקים')}</span>
          <span className="asst-effect-flow" dir="ltr">{breaks.flow || '-'}</span>
          <span className="asst-effect-delta" dir="ltr">{breaks.delta}</span>
        </div>
      ) : null}
    </div>
  );
}

// Bulk clarity: a batch that carries many changes, or that touches activation
// or money levers, gets a quiet banner naming the scale before approval. The
// change count is real (settings fields plus one per other item), and the
// sensitive test only inspects fields the batch actually carries.
const SENSITIVE_FIELD = /activ|enabl|price|cpp|rate|premium|multiplier|budget/i;

function bulkFacts(items) {
  const pending = items.filter((item) => item.status === 'pending');
  let changeCount = 0;
  let sensitive = false;
  for (const item of pending) {
    const rows = item.kind === 'settings_change' ? settingsRows(item.payload) : [];
    changeCount += rows.length > 0 ? rows.length : 1;
    if (item.kind === 'pricing_change') sensitive = true;
    if (rows.some((row) => SENSITIVE_FIELD.test(String(row.field || '')))) sensitive = true;
    if (Array.isArray(item.diff) && item.diff.some((row) => row && SENSITIVE_FIELD.test(String(row.field || '')))) sensitive = true;
  }
  return { changeCount, sensitive };
}

export default function AssistantProposalCard({ batch, locale, busy, applyResult, onApply, onReject, onShowRestore }) {
  const items = Array.isArray(batch.items) ? batch.items : [];
  const pendingIds = items.filter((item) => item.status === 'pending' && item.id).map((item) => item.id);
  const [checked, setChecked] = useState(() => new Set(pendingIds));
  const [confirming, setConfirming] = useState(false);
  const selectedIds = pendingIds.filter((id) => checked.has(id));
  const allSelected = pendingIds.length > 0 && selectedIds.length === pendingIds.length;

  function toggleItem(id) {
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleAll() {
    setChecked(allSelected ? new Set() : new Set(pendingIds));
  }

  async function confirmApply() {
    setConfirming(false);
    await onApply(selectedIds);
  }

  const results = applyResult && Array.isArray(applyResult.results) ? applyResult.results : [];
  const appliedCount = results.filter((row) => row && row.status === 'applied').length;
  const failedCount = results.filter((row) => row && row.status === 'failed').length;
  const jobIds = results.map((row) => row && row.job_id).filter(Boolean);
  const bulk = bulkFacts(items);
  const showBulkBanner = pendingIds.length > 0 && (bulk.changeCount > 3 || bulk.sensitive);
  const bulkText = bulk.changeCount > 3
    ? (bulk.sensitive
      ? pageText(locale, `Broad action, ${bulk.changeCount} changes, including activation or pricing levers.`, `פעולה רחבה, ${bulk.changeCount} שינויים, כולל שינויי הפעלה או תמחור.`)
      : pageText(locale, `Broad action, ${bulk.changeCount} changes.`, `פעולה רחבה, ${bulk.changeCount} שינויים.`))
    : pageText(locale, 'This action touches activation or pricing levers.', 'הפעולה נוגעת בשינויי הפעלה או תמחור.');
  const confirmText = selectedIds.length === 1
    ? pageText(locale, 'One selected action will be applied to the saved data. A restore point is created automatically before applying, so the previous state can be recovered.', 'תוחל פעולה נבחרת אחת על הנתונים השמורים. לפני ההחלה נוצרת אוטומטית נקודת שחזור, כך שאפשר לחזור למצב הקודם.')
    : pageText(locale, `${selectedIds.length} selected actions will be applied to the saved data. A restore point is created automatically before applying, so the previous state can be recovered.`, `יוחלו ${selectedIds.length} פעולות נבחרות על הנתונים השמורים. לפני ההחלה נוצרת אוטומטית נקודת שחזור, כך שאפשר לחזור למצב הקודם.`);

  return (
    <div className="asst-proposal">
      <div className="asst-proposal-head">
        <span>{pageText(locale, 'Proposed actions', 'פעולות מוצעות')}</span>
        <code dir="ltr">{String(batch.batch_id).slice(0, 8)}</code>
      </div>

      {showBulkBanner ? (
        <div className="asst-bulk" role="note">
          <TriangleAlert size={13} />
          <span dir="auto">{bulkText}</span>
        </div>
      ) : null}

      {items.length === 0 ? (
        <p className="asst-item-reason asst-pad">{pageText(locale, 'This batch contains no actions.', 'האצווה הזו אינה מכילה פעולות.')}</p>
      ) : items.map((item) => {
        const kind = KINDS[item.kind] || null;
        const KindIcon = kind ? kind.Icon : Layers;
        const statusPair = STATUS_LABELS[item.status] || null;
        const statusClass = STATUS_LABELS[item.status] ? item.status : 'unknown';
        return (
          <div className={`asst-item${item.status === 'rejected' ? ' rejected' : ''}`} key={item.key || item.id}>
            <div className="asst-item-check">
              {item.status === 'pending' && item.id ? (
                <input type="checkbox" checked={checked.has(item.id)} onChange={() => toggleItem(item.id)} disabled={busy} aria-label={pageText(locale, 'Select this action', 'בחירת הפעולה הזו')} />
              ) : null}
            </div>
            <div className="asst-item-body">
              <div className="asst-item-head">
                <span className="asst-kind"><KindIcon size={12} />{kind ? pageText(locale, kind.en, kind.he) : <code dir="ltr">{item.kind || '?'}</code>}</span>
                <span className={`asst-status-chip ${statusClass}`}>{statusPair ? pageText(locale, statusPair[0], statusPair[1]) : <code dir="ltr">{item.status}</code>}</span>
              </div>
              {item.summary ? <p className="asst-item-summary" dir="auto">{item.summary}</p> : null}
              {item.reason ? <p className="asst-item-reason" dir="auto">{item.reason}</p> : null}
              {!item.summary && !item.reason && !item.payload ? (
                <p className="asst-item-reason">{pageText(locale, 'No details were provided for this action.', 'לא סופקו פרטים לפעולה הזו.')}</p>
              ) : null}
              {Array.isArray(item.diff) && item.diff.length ? <DiffView item={item} locale={locale} /> : <PayloadView item={item} locale={locale} />}
              {item.kind === 'settings_change' && item.effect ? <EffectView effect={item.effect} locale={locale} /> : null}
              {item.status === 'failed' && item.error ? (
                <p className="asst-item-error"><TriangleAlert size={12} /><span dir="auto">{item.error}</span></p>
              ) : null}
            </div>
          </div>
        );
      })}

      {pendingIds.length > 0 && !confirming ? (
        <div className="asst-batch-bar">
          <Button variant="contained" size="small" disabled={busy || selectedIds.length === 0} onClick={() => setConfirming(true)}>
            {pageText(locale, `Approve selected (${selectedIds.length})`, `אשר נבחרים (${selectedIds.length})`)}
          </Button>
          <Button variant="outlined" size="small" disabled={busy || selectedIds.length === 0} onClick={() => onReject(selectedIds)}>
            {pageText(locale, 'Reject selected', 'דחה נבחרים')}
          </Button>
          <label className="asst-select-all">
            <input type="checkbox" checked={allSelected} onChange={toggleAll} disabled={busy} />
            {pageText(locale, 'Select all', 'בחירת הכל')}
          </label>
          {busy ? <span className="asst-busy-note">{pageText(locale, 'Working', 'מבצע')}</span> : null}
        </div>
      ) : null}

      {pendingIds.length > 0 && confirming ? (
        <div className="asst-confirm" role="alertdialog">
          <p>{confirmText}</p>
          <div className="asst-confirm-actions">
            <Button variant="contained" size="small" disabled={busy || selectedIds.length === 0} onClick={confirmApply}>
              {pageText(locale, 'Apply now', 'החל עכשיו')}
            </Button>
            <Button variant="outlined" size="small" disabled={busy} onClick={() => setConfirming(false)}>
              {pageText(locale, 'Cancel', 'ביטול')}
            </Button>
          </div>
        </div>
      ) : null}

      {applyResult ? (
        <div className="asst-result">
          <span>
            {results.length === 1
              ? (appliedCount === 1 ? pageText(locale, 'The action was applied.', 'הפעולה הוחלה.') : pageText(locale, 'The action was not applied.', 'הפעולה לא הוחלה.'))
              : pageText(locale, `Applied ${appliedCount} of ${results.length} actions.`, `הוחלו ${appliedCount} מתוך ${results.length} פעולות.`)}
            {results.length > 1 && failedCount ? ` ${failedCount === 1 ? pageText(locale, 'One failed.', 'אחת נכשלה.') : pageText(locale, `${failedCount} failed.`, `${failedCount} נכשלו.`)}` : ''}
          </span>
          {jobIds.length ? (
            <span className="asst-result-job">{pageText(locale, 'Background job started', 'הופעל תהליך רקע')} <code dir="ltr">{jobIds.join(', ')}</code></span>
          ) : null}
          {applyResult.restoreId ? (
            <button type="button" className="asst-restore-chip" onClick={onShowRestore}>
              <RotateCcw size={12} />
              {pageText(locale, 'A restore point was created', 'נוצרה נקודת שחזור')}
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
