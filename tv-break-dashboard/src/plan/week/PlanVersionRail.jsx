import React from 'react';
import { TextField } from '@mui/material';
import { Archive, GitCompareArrows, History, LockKeyhole, Play, RotateCcw, ShieldCheck } from 'lucide-react';
import { Button } from '../../studio/actions';
import { Figure, Name } from '../../shell/bidi';
import { formatCurrency, formatNumber, pageText } from '../../shell/format';
import { formatStamp } from '../../shell/dates';
import { collapseWarning, diffReason } from './plan-week-model';

function RecordMetric({ label, value }) {
  return (
    <div className="plan-record-metric">
      <span>{label}</span>
      <strong><Figure>{value}</Figure></strong>
    </div>
  );
}

function CheckpointRow({ version, locale, current, selected, canEdit, canEditReason, onCompare, onRestore }) {
  const owned = version.summary?.owned || {};
  return (
    <li className={`plan-checkpoint${selected ? ' is-selected' : ''}${current ? ' is-current' : ''}`}>
      <div className="plan-checkpoint-head">
        <div>
          <strong><Name>{version.name || version.version_id}</Name></strong>
          <span><Figure>{formatStamp(version.created_at) || version.created_at}</Figure></span>
        </div>
        {current && (
          <span className="plan-record-badge is-frozen">
            <ShieldCheck size={13} aria-hidden="true" />
            {pageText(locale, 'Plan of record', 'תוכנית הייחוס')}
          </span>
        )}
      </div>
      <div className="plan-checkpoint-figures">
        <span><Figure>{formatCurrency(owned.revenue, locale)}</Figure></span>
        <span><Figure>{formatNumber(owned.breaks, locale)}</Figure> {pageText(locale, 'breaks', 'ברייקים')}</span>
      </div>
      <div className="plan-checkpoint-actions">
        <Button type="button" variant="outlined" onClick={() => onCompare(version.version_id)}>
          <GitCompareArrows size={14} aria-hidden="true" />
          {pageText(locale, 'Compare to current', 'השוואה לנוכחית')}
        </Button>
        <Button
          type="button"
          variant="outlined"
          disabled={!canEdit || current}
          title={!canEdit ? canEditReason : undefined}
          onClick={() => onRestore(version)}
        >
          <RotateCcw size={14} aria-hidden="true" />
          {current
            ? pageText(locale, 'Already adopted', 'כבר אומצה')
            : pageText(locale, 'Review adoption', 'בדיקה ואימוץ')}
        </Button>
      </div>
    </li>
  );
}

function LiveDiff({ diff, locale }) {
  if (!diff) return null;
  if (diff.available === false) {
    return <p className="plan-workbench-note is-warning" role="status">{diffReason(diff, locale)}</p>;
  }
  if (!diff.available) return null;
  return (
    <div className="plan-live-diff" role="status">
      <span>{pageText(locale, 'Frozen checkpoint against the current plan of record', 'נקודת הבקרה הקפואה מול תוכנית הייחוס הנוכחית')}</span>
      <strong><Name>{diff.version_name}</Name></strong>
      {diff.identical ? (
        <p>{pageText(locale, 'The two weekly plan files are identical.', 'שני קובצי התוכנית השבועית זהים.')}</p>
      ) : (
        <div className="plan-live-diff-figures">
          <RecordMetric label={pageText(locale, 'Revenue delta', 'פער הכנסה')} value={formatCurrency(diff.delta?.revenue, locale)} />
          <RecordMetric label={pageText(locale, 'Break delta', 'פער ברייקים')} value={formatNumber(diff.delta?.breaks, locale)} />
          <RecordMetric label={pageText(locale, 'Days changed', 'ימים שהשתנו')} value={formatNumber(diff.changed_days?.length || 0, locale)} />
        </div>
      )}
    </div>
  );
}

export default function PlanVersionRail({
  locale,
  versions,
  live,
  freshness,
  canEdit,
  canEditReason,
  name,
  note,
  publishState,
  publishError,
  selectedId,
  diff,
  runState,
  runDisabled,
  runDisabledReason,
  onNameChange,
  onNoteChange,
  onPublish,
  onCompare,
  onRestore,
  onRun,
  onOpenHistory,
}) {
  const frozenId = live?.frozen_as || null;
  const frozenVersion = frozenId ? versions.find((version) => version.version_id === frozenId) : null;
  const immutable = Boolean(frozenId);
  const collapse = collapseWarning(live);
  const running = runState === 'running';
  const stale = freshness?.status === 'stale';
  const checkpoints = versions.slice(0, 4);
  const recordRevenue = live?.summary?.owned?.revenue;
  const recordBreaks = live?.summary?.owned?.breaks;
  const recordTitle = immutable
    ? (frozenVersion?.name || frozenId)
    : pageText(locale, 'Latest optimizer run / plan of record', 'הרצת האופטימייזר האחרונה / תוכנית הייחוס');

  return (
    <aside className="card plan-version-rail" aria-labelledby="plan-version-rail-title">
      <div className="plan-version-rail-head">
        <span className="plan-workbench-kicker">{pageText(locale, 'Plan lineage', 'שושלת התוכנית')}</span>
        <h3 id="plan-version-rail-title">{pageText(locale, 'Record & checkpoints', 'תוכנית ייחוס ונקודות בקרה')}</h3>
      </div>

      <section className="card plan-record" aria-label={pageText(locale, 'Current plan record', 'תוכנית הייחוס הנוכחית')}>
        <div className="plan-record-title">
          <span className={`plan-record-badge${immutable ? ' is-frozen' : ''}`}>
            {immutable ? <ShieldCheck size={13} aria-hidden="true" /> : <History size={13} aria-hidden="true" />}
            {immutable
              ? pageText(locale, 'Immutable frozen baseline', 'בסיס קפוא ובלתי־משתנה')
              : pageText(locale, 'Mutable plan of record', 'תוכנית ייחוס ניתנת לשינוי')}
          </span>
          <strong><Name>{recordTitle}</Name></strong>
          <span><Figure>{formatStamp(immutable ? frozenVersion?.created_at : live?.computed_at) || pageText(locale, 'Timestamp unavailable', 'חותמת זמן אינה זמינה')}</Figure></span>
        </div>
        <div className="plan-record-figures">
          <RecordMetric label={pageText(locale, 'Weekly revenue', 'הכנסה שבועית')} value={formatCurrency(recordRevenue, locale)} />
          <RecordMetric label={pageText(locale, 'Weekly breaks', 'ברייקים בשבוע')} value={formatNumber(recordBreaks, locale)} />
        </div>
      </section>

      <div className="plan-workflow-ledger" aria-label={pageText(locale, 'How the working plan becomes a checkpoint', 'איך תוכנית העבודה הופכת לנקודת בקרה')}>
        <p><b>1</b>{pageText(locale, 'Save day placements as reviewed constraints.', 'שומרים שיבוצי יום כמגבלות שנבדקו.')}</p>
        <p><b>2</b>{pageText(locale, 'Run the weekly optimizer to write them into the plan of record.', 'מריצים את האופטימייזר השבועי כדי לכתוב אותן לתוכנית הייחוס.')}</p>
        <p><b>3</b>{pageText(locale, 'Freeze that exact weekly file under a name.', 'מקפיאים את קובץ השבוע המדויק תחת שם.')}</p>
      </div>

      {stale && (
        <div className="plan-workbench-note is-warning">
          <p>{pageText(locale, 'Saved inputs changed after this plan was calculated. A new run is required before they appear in the plan of record.', 'קלטים שמורים השתנו לאחר חישוב התוכנית הזאת. נדרשת הרצה חדשה לפני שיופיעו בתוכנית הייחוס.')}</p>
          <Button type="button" variant="outlined" disabled={running || runDisabled} title={runDisabledReason || undefined} onClick={onRun}>
            <Play size={14} aria-hidden="true" />
            {running ? pageText(locale, 'Optimizer running', 'האופטימייזר פועל') : pageText(locale, 'Review a new weekly run', 'בדיקת הרצה שבועית חדשה')}
          </Button>
        </div>
      )}

      {!immutable && live?.exists !== false && (
        <div className="plan-baseline-setup">
          <div>
            <strong>{pageText(locale, 'Name the current record', 'תנו שם לתוכנית הייחוס')}</strong>
            <p>{pageText(locale, 'This freezes the weekly file on disk. Unrun day edits are not included.', 'הפעולה מקפיאה את קובץ השבוע שעל הדיסק. עריכות יום שטרם הורצו אינן נכללות.')}</p>
          </div>
          <TextField
            size="small"
            label={pageText(locale, 'Baseline name', 'שם הבסיס')}
            value={name}
            onChange={(event) => onNameChange(event.target.value)}
            disabled={!canEdit}
            slotProps={{ htmlInput: { maxLength: 120, dir: 'auto' } }}
          />
          <TextField
            size="small"
            label={pageText(locale, 'Reason or handoff note', 'סיבה או הערת מסירה')}
            value={note}
            onChange={(event) => onNoteChange(event.target.value)}
            disabled={!canEdit}
            slotProps={{ htmlInput: { maxLength: 400, dir: 'auto' } }}
          />
          <Button
            type="button"
            variant="contained"
            disabled={!canEdit || !name.trim() || publishState === 'running'}
            title={!canEdit ? canEditReason : undefined}
            onClick={() => (collapse.collapsed ? onOpenHistory() : onPublish(false))}
          >
            <LockKeyhole size={14} aria-hidden="true" />
            {collapse.collapsed
              ? pageText(locale, 'Review baseline safeguards', 'בדיקת הגנות ההקפאה')
              : pageText(locale, 'Freeze current plan as baseline', 'הקפאת התוכנית הנוכחית כבסיס')}
          </Button>
          {publishError && <p className="plan-workbench-note is-error" role="alert"><Name>{publishError}</Name></p>}
        </div>
      )}

      <section className="plan-checkpoints" aria-labelledby="plan-checkpoints-title">
        <div className="plan-checkpoints-head">
          <h4 id="plan-checkpoints-title">{pageText(locale, 'Frozen checkpoints', 'נקודות בקרה קפואות')}</h4>
          <span><Figure>{formatNumber(versions.length, locale)}</Figure></span>
        </div>
        {checkpoints.length ? (
          <ol>
            {checkpoints.map((version) => (
              <CheckpointRow
                key={version.version_id}
                version={version}
                locale={locale}
                current={version.version_id === frozenId}
                selected={version.version_id === selectedId}
                canEdit={canEdit}
                canEditReason={canEditReason}
                onCompare={onCompare}
                onRestore={onRestore}
              />
            ))}
          </ol>
        ) : (
          <p className="plan-workbench-note">{pageText(locale, 'No immutable checkpoint exists yet.', 'עדיין אין נקודת בקרה בלתי־משתנה.')}</p>
        )}
      </section>

      <LiveDiff diff={diff} locale={locale} />

      <Button className="plan-open-history" type="button" variant="outlined" onClick={onOpenHistory}>
        <Archive size={14} aria-hidden="true" />
        {pageText(locale, 'Open complete version history', 'פתיחת היסטוריית הגרסאות המלאה')}
      </Button>
    </aside>
  );
}
