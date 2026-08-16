import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TextField } from '@mui/material';
import {
  AlertTriangle,
  GitCompareArrows,
  HardDriveDownload,
  LockKeyhole,
  RotateCcw,
  Save,
  ServerCog,
  ShieldCheck,
  Trash2,
} from 'lucide-react';
import { Button } from '../../studio/actions';
import { Figure, Name } from '../../shell/bidi';
import { formatCurrency, formatNumber, pageText } from '../../shell/format';
import { formatStamp } from '../../shell/dates';
import { movesFrom } from '../day/day-board-model';
import { scoreDay } from '../day/day-board-actions';
import {
  changedEditsBetween,
  createVariantScope,
  ensureLocalBaseline,
  loadDayDrafts,
  removeLocalVariant,
  sameEdits,
  sameVariantScope,
  storeLocalVariant,
  variantIsCurrent,
} from './local-plan-variants';
import './local-plan-variants.css';

function VariantMetric({ label, value, tone = '' }) {
  return (
    <div className={`local-variant-metric${tone ? ` is-${tone}` : ''}`}>
      <span>{label}</span>
      <strong><Figure>{value}</Figure></strong>
    </div>
  );
}

function signedCurrency(value, locale) {
  const amount = Number(value || 0);
  if (Math.abs(amount) < 0.005) return formatCurrency(0, locale);
  return `${amount > 0 ? '+' : ''}${formatCurrency(amount, locale)}`;
}

function deltaTone(value) {
  const amount = Number(value || 0);
  if (Math.abs(amount) < 0.005) return 'flat';
  return amount > 0 ? 'gain' : 'loss';
}

function VariantRow({ variant, locale, current, active, compared, deleteArmed, onAdopt, onCompare, onDelete }) {
  const compliant = variant.compliance?.compliant !== false;
  return (
    <li className={`local-variant-row${current ? '' : ' is-stale'}${active ? ' is-active' : ''}`}>
      <div className="local-variant-row-head">
        <div>
          <strong><Name>{variant.name}</Name></strong>
          <span><Figure>{formatStamp(variant.savedAt)}</Figure></span>
        </div>
        <span className={`local-variant-state${current ? '' : ' is-stale'}`}>
          {current
            ? pageText(locale, 'Exact plan identity', 'זהות תוכנית מדויקת')
            : pageText(locale, 'Different plan identity', 'זהות תוכנית אחרת')}
        </span>
      </div>
      <div className="local-variant-metrics">
        <VariantMetric label={pageText(locale, 'Engine-scored revenue', 'הכנסה לפי המנוע')} value={formatCurrency(variant.totals?.revenue, locale)} />
        <VariantMetric label={pageText(locale, 'Δ optimizer baseline', 'פער מבסיס האופטימייזר')} value={signedCurrency(variant.delta?.revenue, locale)} tone={deltaTone(variant.delta?.revenue)} />
        <VariantMetric label={pageText(locale, 'Manual changes', 'שינויים ידניים')} value={formatNumber(variant.editCount, locale)} />
      </div>
      <p className={`local-variant-compliance${compliant ? ' is-ok' : ' is-bad'}`}>
        {compliant
          ? pageText(locale, 'Compliant in the saved engine score', 'תקין בחישוב המנוע שנשמר')
          : pageText(locale, 'Saved score contains a compliance violation', 'החישוב השמור כולל חריגת ציות')}
      </p>
      {!current && (
        <p className="local-variant-stale-note">
          <AlertTriangle size={13} aria-hidden="true" />
          {pageText(locale, 'Visible for audit only. Adoption and like-for-like comparison are blocked.', 'מוצג לביקורת בלבד. אימוץ והשוואה על אותו בסיס חסומים.')}
        </p>
      )}
      <div className="local-variant-actions">
        <Button type="button" variant="outlined" disabled={!current} onClick={() => onAdopt(variant)}>
          <HardDriveDownload size={14} aria-hidden="true" />
          {active ? pageText(locale, 'Open in board', 'פתוחה בלוח') : pageText(locale, 'Open locally', 'פתיחה מקומית')}
        </Button>
        <Button type="button" variant="outlined" aria-pressed={compared} disabled={!current} onClick={() => onCompare(variant.id)}>
          <GitCompareArrows size={14} aria-hidden="true" />
          {compared ? pageText(locale, 'In comparison', 'בהשוואה') : pageText(locale, 'Compare', 'השוואה')}
        </Button>
        <Button className="local-variant-delete" type="button" variant="text" onClick={() => onDelete(variant)}>
          <Trash2 size={14} aria-hidden="true" />
          {deleteArmed ? pageText(locale, 'Confirm remove', 'אישור הסרה') : pageText(locale, 'Remove', 'הסרה')}
        </Button>
      </div>
    </li>
  );
}

function PairComparison({ variants, locale }) {
  if (variants.length === 0) return null;
  if (variants.length === 1) {
    return (
      <div className="local-pair-compare is-waiting" role="status">
        <GitCompareArrows size={15} aria-hidden="true" />
        <span>{pageText(locale, 'Choose one more current draft for a direct comparison.', 'בחרו עוד טיוטה עדכנית להשוואה ישירה.')}</span>
      </div>
    );
  }
  const [first, second] = variants;
  const revenue = Number(second.totals?.revenue || 0) - Number(first.totals?.revenue || 0);
  const breaks = Number(second.totals?.breaks || 0) - Number(first.totals?.breaks || 0);
  const changed = changedEditsBetween(first.edits, second.edits);
  return (
    <section className="local-pair-compare" aria-label={pageText(locale, 'Direct manual variant comparison', 'השוואה ישירה בין גרסאות ידניות')}>
      <div className="local-pair-title">
        <GitCompareArrows size={15} aria-hidden="true" />
        <span><Name>{second.name}</Name> {pageText(locale, 'against', 'מול')} <Name>{first.name}</Name></span>
      </div>
      <div className="local-pair-metrics">
        <VariantMetric label={pageText(locale, 'Revenue difference', 'פער הכנסה')} value={signedCurrency(revenue, locale)} tone={deltaTone(revenue)} />
        <VariantMetric label={pageText(locale, 'Break difference', 'פער ברייקים')} value={formatNumber(breaks, locale)} />
        <VariantMetric label={pageText(locale, 'Placements that differ', 'מיקומים שונים')} value={formatNumber(changed, locale)} />
      </div>
    </section>
  );
}

export default function LocalPlanVariants({ board, live, freshness, workState, locale, notify, onApplyDraft }) {
  const scope = useMemo(() => createVariantScope(board, live, freshness), [board, live, freshness]);
  const scopeToken = [scope.channel, scope.day, scope.planIdentity, scope.computedAt, scope.fingerprint].join('|');
  const [records, setRecords] = useState({ baselines: [], variants: [], error: null });
  const [name, setName] = useState('');
  const [saveState, setSaveState] = useState('idle');
  const [message, setMessage] = useState('');
  const [compareIds, setCompareIds] = useState([]);
  const [deleteArmed, setDeleteArmed] = useState('');
  const scopeRef = useRef(scope);
  const workEditsRef = useRef(workState?.edits || {});
  scopeRef.current = scope;
  workEditsRef.current = workState?.edits || {};

  const reload = useCallback(() => setRecords(loadDayDrafts(scopeRef.current)), []);

  useEffect(() => {
    setName('');
    setCompareIds([]);
    setDeleteArmed('');
    setMessage('');
    if (scope.verifiable && board?.available) ensureLocalBaseline(scope, board);
    setRecords(loadDayDrafts(scope));
  }, [scopeToken, board]);

  const currentBaseline = records.baselines.find((item) => sameVariantScope(item.scope, scope)) || null;
  const currentVariants = records.variants.filter((item) => variantIsCurrent(item, scope));
  const editCount = Object.keys(workState?.edits || {}).length;
  const activeVariant = currentVariants.find((item) => sameEdits(item.edits, workState?.edits || {})) || null;
  const compared = compareIds.map((id) => currentVariants.find((item) => item.id === id)).filter(Boolean);

  const saveExactDraft = useCallback(async () => {
    const cleanName = name.trim();
    if (!scope.verifiable) {
      setMessage(pageText(locale, 'The plan file identity or run timestamp is unavailable, so a persistent draft cannot be verified.', 'זהות קובץ התוכנית או חותמת ההרצה אינן זמינות, ולכן אי אפשר לאמת טיוטה מתמשכת.'));
      return;
    }
    if (!cleanName || editCount === 0) {
      setMessage(pageText(locale, 'Name the variant after making at least one manual change.', 'תנו שם לגרסה לאחר ביצוע שינוי ידני אחד לפחות.'));
      return;
    }
    if (currentVariants.some((item) => item.name.toLocaleLowerCase() === cleanName.toLocaleLowerCase())) {
      setMessage(pageText(locale, 'A current draft already uses this name.', 'כבר קיימת טיוטה עדכנית בשם הזה.'));
      return;
    }
    const measuredScope = scope;
    const measuredEdits = JSON.parse(JSON.stringify(workState?.edits || {}));
    setSaveState('measuring');
    setMessage('');
    try {
      const score = await scoreDay(scope.day, movesFrom(measuredEdits));
      if (!sameVariantScope(measuredScope, scopeRef.current) || !sameEdits(measuredEdits, workEditsRef.current)) {
        setMessage(pageText(locale, 'The plan or working arrangement changed during measurement. Measure and save again.', 'התוכנית או סידור העבודה השתנו במהלך המדידה. מדדו ושמרו שוב.'));
        return;
      }
      const result = storeLocalVariant(scope, board, {
        name: cleanName,
        edits: measuredEdits,
        optimizerTotals: score.saved,
        totals: score.current,
        delta: score.delta,
        compliance: score.compliance,
      });
      if (!result.ok) throw new Error(result.error);
      setName('');
      reload();
      notify?.(
        'The exact arrangement was saved in this browser only. The plan of record was not changed.',
        'הסידור המדויק נשמר בדפדפן הזה בלבד. תוכנית הייחוס לא השתנתה.',
      );
    } catch (error) {
      setMessage(error.message);
    } finally {
      setSaveState('idle');
    }
  }, [board, currentVariants, editCount, locale, name, notify, reload, scope, workState?.edits]);

  const adopt = useCallback((variant) => {
    if (!variantIsCurrent(variant, scope)) return;
    onApplyDraft?.({
      id: `adopt:${variant.id}:${Date.now()}`,
      kind: 'browser-local-variant',
      channel: scope.channel,
      day: scope.day,
      scope,
      edits: variant.edits,
    });
    notify?.(
      'The browser draft is open on the board. Nothing was written to the server.',
      'טיוטת הדפדפן פתוחה בלוח. דבר לא נכתב לשרת.',
    );
  }, [locale, notify, onApplyDraft, scope]);

  const revert = useCallback(() => {
    onApplyDraft?.({
      id: `baseline:${scope.fingerprint}:${Date.now()}`,
      kind: 'optimizer-baseline',
      channel: scope.channel,
      day: scope.day,
      scope,
      edits: {},
    });
    notify?.(
      'The working board returned to the optimizer arrangement captured for this exact plan. Nothing was written.',
      'לוח העבודה חזר לסידור האופטימייזר שנלכד עבור התוכנית המדויקת הזאת. דבר לא נכתב.',
    );
  }, [notify, onApplyDraft, scope]);

  const toggleCompare = useCallback((id) => {
    setCompareIds((current) => {
      if (current.includes(id)) return current.filter((item) => item !== id);
      return current.length < 2 ? [...current, id] : [current[1], id];
    });
  }, []);

  const remove = useCallback((variant) => {
    if (deleteArmed !== variant.id) {
      setDeleteArmed(variant.id);
      return;
    }
    const result = removeLocalVariant(variant);
    if (!result.ok) setMessage(result.error);
    setCompareIds((current) => current.filter((id) => id !== variant.id));
    setDeleteArmed('');
    reload();
  }, [deleteArmed, reload]);

  const goToServerReview = useCallback(() => {
    const control = document.getElementById('day-board-server-replan');
    control?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    window.setTimeout(() => control?.focus(), 250);
  }, []);

  return (
    <aside className="card local-variants" aria-labelledby="local-variants-title">
      <div className="local-variants-head">
        <div>
          <span className="plan-workbench-kicker">{pageText(locale, 'Manual variants', 'גרסאות ידניות')}</span>
          <h3 id="local-variants-title">{pageText(locale, 'Exact drafts, before any write', 'טיוטות מדויקות, לפני כל כתיבה')}</h3>
        </div>
        <span className="local-only-badge"><HardDriveDownload size={13} />{pageText(locale, 'This browser', 'בדפדפן הזה')}</span>
      </div>

      <div className={`local-baseline${currentBaseline ? ' is-ready' : ' is-unverified'}`}>
        {currentBaseline ? <ShieldCheck size={16} aria-hidden="true" /> : <AlertTriangle size={16} aria-hidden="true" />}
        <div>
          <strong>{currentBaseline
            ? pageText(locale, 'Immutable optimizer baseline captured', 'בסיס אופטימייזר בלתי־משתנה נלכד')
            : pageText(locale, 'Verified baseline unavailable', 'בסיס מאומת אינו זמין')}</strong>
          <span>{currentBaseline
            ? `${formatStamp(currentBaseline.capturedAt)} · ${formatCurrency(currentBaseline.totals?.revenue, locale)}`
            : pageText(locale, 'Local adoption stays blocked until the plan identity and run timestamp are known.', 'אימוץ מקומי נשאר חסום עד שזהות התוכנית וחותמת ההרצה ידועות.')}</span>
        </div>
        {records.baselines.length > (currentBaseline ? 1 : 0) && (
          <small>{formatNumber(records.baselines.length - (currentBaseline ? 1 : 0), locale)} {pageText(locale, 'earlier baseline records retained', 'רשומות בסיס קודמות נשמרו')}</small>
        )}
      </div>

      <div className="local-write-choice">
        <section>
          <span className="local-choice-number">1</span>
          <div>
            <strong>{pageText(locale, 'Keep the exact arrangement', 'שמירת הסידור המדויק')}</strong>
            <p>{pageText(locale, 'Names this manual state in localStorage. No API call writes it; the plan of record does not move.', 'נותן שם למצב הידני ב־localStorage. אף קריאת API אינה כותבת אותו; תוכנית הייחוס אינה משתנה.')}</p>
          </div>
        </section>
        <div className="local-draft-form">
          <TextField
            size="small"
            label={pageText(locale, 'Variant name', 'שם הגרסה')}
            value={name}
            onChange={(event) => { setName(event.target.value); setMessage(''); }}
            disabled={!scope.verifiable || saveState === 'measuring'}
            slotProps={{ htmlInput: { maxLength: 80, dir: 'auto' } }}
          />
          <Button type="button" variant="contained" disabled={!scope.verifiable || !name.trim() || editCount === 0 || saveState === 'measuring'} onClick={saveExactDraft}>
            <Save size={14} aria-hidden="true" />
            {saveState === 'measuring'
              ? pageText(locale, 'Measuring with engine', 'מודד באמצעות המנוע')
              : pageText(locale, `Keep exact (${editCount})`, `שמירה מדויקת (${editCount})`)}
          </Button>
        </div>
        <section>
          <span className="local-choice-number">2</span>
          <div>
            <strong>{pageText(locale, 'Change the official plan', 'שינוי התוכנית הרשמית')}</strong>
            <p>{pageText(locale, 'The reviewed server action writes placement constraints, then re-runs the optimizer. Other ids, counts and placements may change.', 'פעולת השרת הנבדקת כותבת מגבלות מיקום ואז מריצה מחדש את האופטימייזר. מזהים, כמויות ומיקומים אחרים עשויים להשתנות.')}</p>
          </div>
          <Button type="button" variant="outlined" disabled={editCount === 0} onClick={goToServerReview}>
            <ServerCog size={14} aria-hidden="true" />
            {pageText(locale, 'Go to reviewed re-plan', 'מעבר לתכנון מחדש הנבדק')}
          </Button>
        </section>
      </div>

      <div className="local-variant-working">
        <span>{activeVariant
          ? pageText(locale, `Working from “${activeVariant.name}”`, `עובדים מתוך „${activeVariant.name}”`)
          : editCount > 0
            ? pageText(locale, 'Unsaved manual working state', 'מצב עבודה ידני שטרם נשמר')
            : pageText(locale, 'At the captured optimizer baseline', 'בבסיס האופטימייזר שנלכד')}</span>
        <Button type="button" variant="outlined" disabled={!currentBaseline || editCount === 0} onClick={revert}>
          <RotateCcw size={14} aria-hidden="true" />
          {pageText(locale, 'Revert working board', 'חזרה לבסיס העבודה')}
        </Button>
      </div>

      {message && <p className="local-variant-message" role="alert"><AlertTriangle size={14} /><Name>{message}</Name></p>}
      {records.error && <p className="local-variant-message" role="alert"><LockKeyhole size={14} /><Name>{records.error}</Name></p>}
      <PairComparison variants={compared} locale={locale} />

      <div className="local-variant-list-head">
        <strong>{pageText(locale, 'Named browser drafts', 'טיוטות דפדפן בעלות שם')}</strong>
        <span><Figure>{formatNumber(records.variants.length, locale)}</Figure></span>
      </div>
      {records.variants.length ? (
        <ol className="local-variant-list">
          {records.variants.map((variant) => (
            <VariantRow
              key={variant.id}
              variant={variant}
              locale={locale}
              current={variantIsCurrent(variant, scope)}
              active={activeVariant?.id === variant.id}
              compared={compareIds.includes(variant.id)}
              deleteArmed={deleteArmed === variant.id}
              onAdopt={adopt}
              onCompare={toggleCompare}
              onDelete={remove}
            />
          ))}
        </ol>
      ) : (
        <p className="local-variant-empty">{pageText(locale, 'Make a safe manual edit, then name its exact arrangement here.', 'בצעו עריכה ידנית בטוחה ואז תנו כאן שם לסידור המדויק שלה.')}</p>
      )}
    </aside>
  );
}
