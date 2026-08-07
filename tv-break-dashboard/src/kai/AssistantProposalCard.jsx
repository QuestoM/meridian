import React, { useState } from 'react';
import { Button } from '@mui/material';
import { Building2, CalendarClock, ExternalLink, Layers, Lock, PlayCircle, Scale, SlidersHorizontal, Tag, TriangleAlert, Users } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { Figure, Code, Name } from '../shell/bidi';
import AssistantUndo from './AssistantUndo';
import EffectView from './AssistantEffectView';
import ProposalSummary from './AssistantProposalSummary';
import FieldName from './kai-field-name';
import { inApprovedWords } from './kai-vocabulary';

// One proposal batch from the assistant, rendered for explicit approval. The
// card never applies anything by itself: selection, an inline confirm step that
// names the automatic restore point, and the apply and reject calls all go
// through the parent, so the inline chat copy and the pending tab stay in sync.
// Missing fields render as honest fallbacks, and after an apply the restore
// point it created is an undo control on this card rather than a note about a
// page somewhere else.

// Keyed on the kind the server actually sends, which is assistant_tools.py's
// KIND_BY_TOOL and nothing else. Keying on a kind the server never emits prints
// the raw key beside a card that has a measured before and after, and then
// denies that it has one, which is what this map did for every settings and
// every rate-card proposal until it was measured.
export const KINDS = {
  settings: { Icon: SlidersHorizontal, en: 'Settings change', he: 'שינוי הגדרות' },
  constraint: { Icon: Lock, en: 'New restriction', he: 'הגבלה חדשה' },
  override: { Icon: Layers, en: 'Pin', he: 'נעיצה' },
  pricing: { Icon: Tag, en: 'Rate-card change', he: 'שינוי תמחור' },
  advertiser_change: { Icon: Users, en: 'Advertiser change', he: 'שינוי מפרסם' },
  recompute: { Icon: PlayCircle, en: 'Run the plan', he: 'הרצת התוכנית' },
  event_change: { Icon: CalendarClock, en: 'Calendar event change', he: 'שינוי אירוע ביומן' },
  agency_change: { Icon: Building2, en: 'Agency change', he: 'שינוי סוכנות' },
  agency_link_change: { Icon: Building2, en: 'Agency link change', he: 'שינוי שיוך סוכנות' },
  agency_condition_change: { Icon: Building2, en: 'Agency condition change', he: 'שינוי תנאי סוכנות' },
};

// Which proposal kinds carry a measured before-and-after. Only a settings
// change is simulated against the real optimizer today, so every other kind
// says plainly that a money preview was not computed rather than leaving the
// reader to assume one is missing by accident.
const MEASURED_EFFECT_KINDS = new Set(['settings']);

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

function appliedLabel(iso, locale) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
}

// Four of the fields Kai may propose are the broadcast licence, not ordinary
// settings. The server says so on the item, with the date the limits in force
// took effect and whether this account may change them, and it is printed here
// before the approval rather than discovered after it.
function PermissionView({ permission, locale }) {
  if (!permission) return null;
  const fields = Array.isArray(permission.fields) ? permission.fields : [];
  return (
    <div className={permission.may_change ? 'asst-permission' : 'asst-permission blocked'} role="note">
      <p><Scale size={12} />{pageText(locale, permission.basis_en || '', permission.basis_he || '')}</p>
      {fields.length ? <div className="asst-permission-fields">{fields.map((field) => <Code key={field}>{field}</Code>)}</div> : null}
      {permission.effective_date ? (
        <p>{pageText(locale, 'The limits in force took effect on ', 'המגבלות שבתוקף נכנסו לתוקף ב-')}<Figure>{permission.effective_date}</Figure>{'.'}</p>
      ) : null}
      <p>{pageText(locale, permission.record_en || '', permission.record_he || '')}</p>
      {!permission.may_change && permission.reason ? <p>{String(permission.reason)}</p> : null}
    </div>
  );
}

// A settings payload becomes field rows. A {from,to} or {before,after}
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
  if (item.kind === 'settings') {
    const rows = settingsRows(item.payload);
    if (!rows.length) return null;
    const shown = rows.slice(0, 8);
    return (
      <div className="asst-fields">
        {shown.map((row) => (
          <div className="asst-field-row" key={row.field}>
            <span className="asst-field-name"><Code>{row.field}</Code></span>
            <span className="asst-field-value">
              {row.from !== null && row.from !== undefined
                ? <><Figure>{shortValue(row.from)}</Figure>{' → '}<Figure>{shortValue(row.to)}</Figure></>
                : <Figure>{shortValue(row.to)}</Figure>}
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
          <span className="asst-field-name"><Code>{key}</Code></span>
          <span className="asst-field-value"><Figure>{shortValue(value)}</Figure></span>
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
        <Name>{header}</Name>
        {isCreation ? <span className="asst-pdiff-new">{pageText(locale, 'New', 'חדש')}</span> : null}
      </div>
      <div className="asst-pdiff-grid">
        <div className="asst-pdiff-row head">
          <span>{pageText(locale, 'Field', 'שדה')}</span>
          <span>{pageText(locale, 'Before', 'לפני')}</span>
          <span>{pageText(locale, 'After', 'אחרי')}</span>
        </div>
        {rows.map((row, index) => (
          <div className="asst-pdiff-row" key={index}>
            <span className="asst-pdiff-field"><FieldName name={row.field} /></span>
            <span className="asst-pdiff-before">{row.before === null || row.before === undefined ? '-' : shortValue(row.before)}</span>
            <span className="asst-pdiff-after">{shortValue(row.after)}</span>
          </div>
        ))}
      </div>
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
    const rows = item.kind === 'settings' ? settingsRows(item.payload) : [];
    changeCount += rows.length > 0 ? rows.length : 1;
    if (item.kind === 'pricing') sensitive = true;
    if (rows.some((row) => SENSITIVE_FIELD.test(String(row.field || '')))) sensitive = true;
    if (Array.isArray(item.diff) && item.diff.some((row) => row && SENSITIVE_FIELD.test(String(row.field || '')))) sensitive = true;
  }
  return { changeCount, sensitive };
}

export default function AssistantProposalCard({ batch, locale, busy, applyResult, onApply, onReject, onShowRestore, notify, onUndone }) {
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
  // Every restore point this batch produced, read from the batch itself, so the
  // undo is here on the next visit and not only in the tab that applied it.
  const restorePoints = Array.isArray(batch.restorePoints) ? batch.restorePoints : [];
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
        <Code>{String(batch.batch_id).slice(0, 8)}</Code>
      </div>

      {showBulkBanner ? (
        <div className="asst-bulk" role="note">
          <TriangleAlert size={13} />
          <span>{bulkText}</span>
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
                <span className="asst-kind"><KindIcon size={12} />{kind ? pageText(locale, kind.en, kind.he) : <Code>{item.kind || '?'}</Code>}</span>
                <span className={`asst-status-chip ${statusClass}`}>{statusPair ? pageText(locale, statusPair[0], statusPair[1]) : <Code>{item.status}</Code>}</span>
              </div>
              <ProposalSummary item={item} locale={locale} className="asst-item-summary" />
              {item.reason ? <p className="asst-item-reason">{inApprovedWords(item.reason)}</p> : null}
              {!item.summary && !item.reason && !item.payload ? (
                <p className="asst-item-reason">{pageText(locale, 'No details were provided for this action.', 'לא סופקו פרטים לפעולה הזו.')}</p>
              ) : null}
              {Array.isArray(item.diff) && item.diff.length ? <DiffView item={item} locale={locale} /> : <PayloadView item={item} locale={locale} />}
              {item.kind === 'settings' && item.effect ? <EffectView effect={item.effect} basis={item.effect_basis} locale={locale} /> : null}
              {item.permission ? <PermissionView permission={item.permission} locale={locale} /> : null}
              {item.status === 'pending' && !MEASURED_EFFECT_KINDS.has(item.kind) ? (
                <p className="asst-effect-note">{pageText(locale, 'A measured before and after is computed for settings changes only. The fields above are exactly what this would write.', 'לפני ואחרי נמדדים מחושבים לשינויי הגדרות בלבד. השדות שלמעלה הם בדיוק מה שייכתב.')}</p>
              ) : null}
              {item.status === 'failed' && item.error ? (
                <p className="asst-item-error">
                  <TriangleAlert size={12} />
                  <span>{pageText(locale, 'The action failed.', 'הפעולה נכשלה.')}</span>
                  <Name>{String(item.error)}</Name>
                </p>
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
            <span className="asst-result-job">{pageText(locale, 'Background job started', 'הופעל תהליך רקע')} <Code>{jobIds.join(', ')}</Code></span>
          ) : null}
        </div>
      ) : null}

      {restorePoints.length ? (
        <div className="asst-restore-block">
          {restorePoints.map((point) => (
            <div className="asst-restore-point" key={point.restoreId}>
              <p className="asst-restore-line">
                {pageText(locale, 'A restore point was created before this change.', 'נוצרה נקודת שחזור לפני השינוי הזה.')}
                {point.appliedAt ? <time><Figure>{appliedLabel(point.appliedAt, locale)}</Figure></time> : null}
                {point.appliedBy ? <span>{pageText(locale, 'Applied by ', 'הוחל על ידי ')}<Name>{point.appliedBy}</Name></span> : null}
              </p>
              <AssistantUndo locale={locale} restoreId={point.restoreId} notify={notify} onDone={onUndone} />
            </div>
          ))}
          <button type="button" className="asst-restore-chip" onClick={() => onShowRestore(restorePoints[restorePoints.length - 1].versionId)}>
            <ExternalLink size={12} />
            {pageText(locale, 'See it in the history', 'הצגה בהיסטוריה')}
          </button>
        </div>
      ) : null}
    </div>
  );
}
