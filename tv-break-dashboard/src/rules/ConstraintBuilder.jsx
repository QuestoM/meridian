import React, { useState } from 'react';
import { FormControl, MenuItem, Select, TextField } from '@mui/material';
import { Button } from '../studio/actions';
import { Save, Send, Trash2 } from 'lucide-react';
import ConsequenceDialog, { focusAfterDialogClose } from '../safety/ConsequenceDialog';
import { GroupNode, defaultGroup } from './constraint-predicate';
import { API_BASE, buildBody, failure, normalizeRows, predicateComplete, rowSentence, t } from './constraint-builder-helpers';
import { EFFECT_LIST, detailWords, effectLabel } from './rules-lib';

// ---- Main ConstraintBuilder export -----------------------------------------
// onGlobalRefresh (optional) is called after a successful save or delete so the
// page-level freshness banner re-reads its verdict; both mutate a fingerprinted
// schedule input.
function ConstraintBuilder({ locale, notify, onRecompute, recomputeState, onGlobalRefresh }) {
  const he = locale === 'he';
  const [hints, setHints] = useState({ programme: [], genre: [], channels: [], available_channels: [] });
  // The value picker offers the operator's own lineup and nothing else. The
  // scope note that comes back with it says which channel was read, and when
  // none is declared it says why the lists are empty and where to declare one.
  const [scope, setScope] = useState(null);
  const [optionsLoaded, setOptionsLoaded] = useState(false);
  const [items, setItems] = useState([]);
  const [sentences, setSentences] = useState(() => new Map());
  const [available, setAvailable] = useState(true);
  const [saving, setSaving] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [preview, setPreview] = useState(null);
  const [previewKey, setPreviewKey] = useState('');
  const [previewError, setPreviewError] = useState('');
  const [deleteReview, setDeleteReview] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const constraintListHeadingRef = React.useRef(null);

  const [draft, setDraft] = useState({
    effect: 'FIX_OFFSET',
    offset_mmss: '00:00',
    offset_min: '00:00',
    offset_max: '00:00',
    pin_count: 1,
    duration_min: 30,
    duration_max: 120,
    order_index: '',
    notes: '',
  });

  const [whereTree, setWhereTree] = useState(defaultGroup);

  React.useEffect(() => {
    let active = true;
    async function load() {
      try {
        const res = await fetch(`${API_BASE}/api/constraints/options`);
        if (res.ok && active) {
          const data = await res.json();
          setHints({
            programme: normalizeRows(data.programmes || data.programme_list),
            genre: normalizeRows(data.genres || data.genre_list),
            channels: normalizeRows(data.channels),
            available_channels: normalizeRows(data.available_channels),
            dayparts: normalizeRows(data.dayparts),
            weekdays: normalizeRows(data.weekdays),
          });
          setScope(data.scope || null);
        }
      } catch {
        // fall through to defaults
      } finally {
        if (active) setOptionsLoaded(true);
      }
      try {
        const listRes = await fetch(`${API_BASE}/api/constraints`);
        if (listRes.status === 404) {
          if (active) setAvailable(false);
          return;
        }
        if (listRes.ok && active) {
          const payload = await listRes.json();
          setItems(normalizeRows(payload));
        }
      } catch {
        // leave list empty
      }
      try {
        const saidRes = await fetch(`${API_BASE}/api/constraints/restrictions`);
        if (saidRes.ok && active) {
          const payload = await saidRes.json();
          const said = new Map();
          (payload.restrictions || []).forEach((record) => {
            said.set(String(record.restriction_id || ''), { en: record.sentence_en || '', he: record.sentence_he || '' });
          });
          setSentences(said);
        }
      } catch {
        // a row with no sentence joined to it reads back its own note
      }
    }
    load();
    return () => { active = false; };
  }, []);

  function updateDraft(field, value) {
    setDraft((current) => ({ ...current, [field]: value }));
    setPreviewKey('');
  }

  function updateWhere(next) {
    setWhereTree(next);
    setPreviewKey('');
  }

  async function previewConstraint() {
    const body = buildBody(draft, whereTree);
    if (!predicateComplete(body.where)) {
      setPreview(null);
      setPreviewError(t(locale, 'Complete every filter condition before measuring this rule.', 'יש להשלים כל תנאי סינון לפני מדידת הכלל.'));
      return;
    }
    setPreviewing(true);
    setPreviewError('');
    try {
      const res = await fetch(`${API_BASE}/api/constraints/effect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw await failure(res);
      const measured = await res.json();
      setPreview(measured);
      setPreviewKey(JSON.stringify(body));
    } catch (err) {
      setPreview(null);
      setPreviewError(detailWords(err, locale));
    } finally {
      setPreviewing(false);
    }
  }

  async function saveConstraint() {
    const body = buildBody(draft, whereTree);
    const currentPreview = previewKey === JSON.stringify(body) && preview;
    if (!currentPreview || Number(currentPreview.summary?.matched_segments || 0) === 0) {
      notify('Measure a complete rule that matches the plan before saving it.', 'יש למדוד כלל שלם שתואם לתוכנית לפני שמירתו.');
      return;
    }
    setSaving(true);
    try {
      const res = await fetch(`${API_BASE}/api/constraints`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (res.status === 404) {
        setAvailable(false);
        notify('The constraints API is not available yet.', 'ממשק האילוצים עדיין לא זמין.');
        return;
      }
      if (!res.ok) throw await failure(res);
      const saved = await res.json();
      const savedId = saved.constraint_id ?? saved.id;
      setItems((current) => [...current, { ...body, id: savedId || `constraint-${current.length + 1}` }]);
      notify('Constraint saved.', 'האילוץ נשמר.');
      setPreviewKey('');
      onGlobalRefresh?.();
    } catch (err) {
      notify(`Saving the constraint failed (${detailWords(err, 'en')}).`, `שמירת האילוץ נכשלה (${detailWords(err, 'he')}).`);
    } finally {
      setSaving(false);
    }
  }

  async function deleteConstraint(id) {
    const matchesId = (item) => (item.constraint_id ?? item.id) === id;
    try {
      const res = await fetch(`${API_BASE}/api/constraints/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (res.status === 404) {
        setItems((current) => current.filter((item) => !matchesId(item)));
        return true;
      }
      if (!res.ok) throw await failure(res);
      setItems((current) => current.filter((item) => !matchesId(item)));
      notify('Constraint removed.', 'האילוץ הוסר.');
      onGlobalRefresh?.();
      return true;
    } catch (err) {
      notify(`Removing the constraint failed (${detailWords(err, 'en')}).`, `הסרת האילוץ נכשלה (${detailWords(err, 'he')}).`);
      return false;
    }
  }

  async function confirmDeleteConstraint() {
    const id = deleteReview && (deleteReview.constraint_id ?? deleteReview.id);
    if (!id) return;
    setDeleting(true);
    const removed = await deleteConstraint(id);
    setDeleting(false);
    if (removed) {
      setDeleteReview(null);
      focusAfterDialogClose(constraintListHeadingRef);
    }
  }

  const { effect } = draft;
  const currentBody = buildBody(draft, whereTree);
  const previewCurrent = previewKey === JSON.stringify(currentBody);
  const matchedSegments = previewCurrent ? Number(preview?.summary?.matched_segments || 0) : 0;

  return (
    <section className="card settings-panel wide constraint-builder">
      <div className="settings-panel-head">
        <div>
          <h2>{t(locale, 'Constraint builder', 'בונה אילוצים')}</h2>
          <p>{t(locale, 'Set the filter conditions, then choose the effect to apply', 'הגדירו את תנאי הסינון ולאחר מכן בחרו את ההשפעה להחלה')}</p>
        </div>
      </div>

      {!available && (
        <p className="constraint-builder-warning">
          {t(locale, 'The constraints API responded with 404. Saving is disabled until it is online.', 'ממשק האילוצים החזיר 404. השמירה מושבתת עד שיהיה זמין.')}
        </p>
      )}

      <div className="cb-section-label">{t(locale, 'When (filter conditions)', 'כאשר (תנאי סינון)')}</div>
      {optionsLoaded && scope && scope.scoped && (
        <p className="cb-scope-note">
          {t(locale, `The suggestions cover the ${hints.programme.length} programmes on ${scope.scope_channel}, the channel this operator owns.`, `ההצעות כוללות את ${hints.programme.length} התוכניות בערוץ ${scope.scope_channel}, הערוץ שבבעלות המפעיל.`)}
        </p>
      )}
      {optionsLoaded && scope && !scope.scoped && (
        <p className="cb-scope-note empty" role="status">
          {t(locale, 'No channel is declared, so there are no programme suggestions. Declare the channel you own under Rules, channel and model.', 'עדיין לא הוצהר ערוץ, ולכן אין הצעות תוכניות. הצהירו על הערוץ שלכם בכללים, ערוץ ומודל.')}
        </p>
      )}
      <div className="cb-tree-root">
        <GroupNode
          group={whereTree}
          onUpdate={updateWhere}
          onDelete={null}
          hints={hints}
          locale={locale}
          level={0}
        />
      </div>

      <div className="cb-section-label cb-effect-label">{t(locale, 'Apply effect', 'אפקט להחלה')}</div>
      <div className="constraint-builder-form">
        <div className="constraint-field">
          <span className="adv-field-label">{t(locale, 'Effect', 'אפקט')}</span>
          <FormControl size="small">
            <Select value={effect} onChange={(e) => updateDraft('effect', e.target.value)}>
              {EFFECT_LIST.map((ef) => (
                <MenuItem key={ef.value} value={ef.value}>{t(locale, ef.label_en, ef.label_he)}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </div>

        {effect === 'FIX_OFFSET' && (
          <div className="constraint-field">
            <span className="adv-field-label">{t(locale, 'Offset (MM:SS)', 'היסט (דק:שנ)')}</span>
            <TextField size="small" value={draft.offset_mmss} onChange={(e) => updateDraft('offset_mmss', e.target.value)} slotProps={{ htmlInput: { dir: 'ltr', placeholder: '02:30' } }} />
          </div>
        )}

        {effect === 'OFFSET_WINDOW' && (
          <>
            <div className="constraint-field">
              <span className="adv-field-label">{t(locale, 'Min offset (MM:SS)', 'היסט מינ (דק:שנ)')}</span>
              <TextField size="small" value={draft.offset_min} onChange={(e) => updateDraft('offset_min', e.target.value)} slotProps={{ htmlInput: { dir: 'ltr' } }} />
            </div>
            <div className="constraint-field">
              <span className="adv-field-label">{t(locale, 'Max offset (MM:SS)', 'היסט מקס (דק:שנ)')}</span>
              <TextField size="small" value={draft.offset_max} onChange={(e) => updateDraft('offset_max', e.target.value)} slotProps={{ htmlInput: { dir: 'ltr' } }} />
            </div>
          </>
        )}

        {effect === 'PIN_COUNT' && (
          <div className="constraint-field">
            <span className="adv-field-label">{t(locale, 'Break count', 'מספר ברייקים')}</span>
            <TextField type="number" size="small" value={draft.pin_count} onChange={(e) => updateDraft('pin_count', e.target.value)} slotProps={{ htmlInput: { min: 0, dir: 'ltr' } }} />
          </div>
        )}

        {effect === 'DURATION_RANGE' && (
          <>
            <div className="constraint-field">
              <span className="adv-field-label">{t(locale, 'Min duration (s)', 'אורך מינ (שנ)')}</span>
              <TextField type="number" size="small" value={draft.duration_min} onChange={(e) => updateDraft('duration_min', e.target.value)} slotProps={{ htmlInput: { min: 0, dir: 'ltr' } }} />
            </div>
            <div className="constraint-field">
              <span className="adv-field-label">{t(locale, 'Max duration (s)', 'אורך מקס (שנ)')}</span>
              <TextField type="number" size="small" value={draft.duration_max} onChange={(e) => updateDraft('duration_max', e.target.value)} slotProps={{ htmlInput: { min: 0, dir: 'ltr' } }} />
            </div>
          </>
        )}

        <div className="constraint-field">
          <span className="adv-field-label">{t(locale, 'Order index (optional)', 'אינדקס סדר (רשות)')}</span>
          <TextField type="number" size="small" value={draft.order_index} onChange={(e) => updateDraft('order_index', e.target.value)} slotProps={{ htmlInput: { min: 0, dir: 'ltr' } }} />
        </div>

        <div className="constraint-field">
          <span className="adv-field-label">{t(locale, 'Notes (optional)', 'הערות (רשות)')}</span>
          <TextField size="small" value={draft.notes} onChange={(e) => updateDraft('notes', e.target.value)} />
        </div>
      </div>

      <div className="constraint-builder-actions">
        <Button type="button" variant="outlined" className="run-button" disabled={previewing || saving || !available} onClick={previewConstraint}>
          {previewing ? t(locale, 'Measuring', 'מודד') : t(locale, 'Measure effect', 'מדידת ההשפעה')}
        </Button>
        <Button type="button" variant="contained" className="run-button" disabled={saving || !available || !previewCurrent || matchedSegments === 0} onClick={saveConstraint}>
          <Save size={14} />
          {t(locale, 'Save constraint', 'שמירת אילוץ')}
        </Button>
        <Button type="button" variant="outlined" className="run-button" disabled={recomputeState === 'running'} onClick={() => onRecompute && onRecompute()}>
          <Send size={14} />
          {t(locale, 'Review weekly run', 'בדיקת ההרצה השבועית')}
        </Button>
      </div>

      {previewError && <p className="constraint-builder-warning" role="alert">{previewError}</p>}
      {previewCurrent && preview && (
        <p className="cb-scope-note" role="status">
          {t(
            locale,
            `Measured without writing: ${matchedSegments} plan segments match; breaks move from ${preview.summary.before_total_breaks} to ${preview.summary.after_total_breaks}, and revenue from ${preview.summary.before_revenue} to ${preview.summary.after_revenue}.`,
            `נמדד ללא כתיבה: ${matchedSegments} רצועות בתוכנית תואמות; מספר הברייקים נע מ־${preview.summary.before_total_breaks} ל־${preview.summary.after_total_breaks}, וההכנסה מ־${preview.summary.before_revenue} ל־${preview.summary.after_revenue}.`,
          )}
        </p>
      )}

      <div className="constraint-list">
        <div className="panel-head">
          <h3 ref={constraintListHeadingRef} tabIndex={-1}>{t(locale, 'Existing constraints', 'אילוצים קיימים')}</h3>
          <span>{items.length}</span>
        </div>
        {items.length === 0 ? (
          <p className="constraint-list-empty">{t(locale, 'No constraints yet.', 'אין אילוצים עדיין.')}</p>
        ) : (
          <ul>
            {items.map((item, index) => {
              const itemId = item.constraint_id ?? item.id;
              return (
                <li className="card" key={itemId ?? `constraint-${index}`}>
                  <span className="constraint-chip">{effectLabel(item.effect, locale)}</span>
                  <span className="constraint-scope">{item.where ? t(locale, 'filter conditions', 'תנאי סינון') : `${item.scope_type}: ${item.scope_value || t(locale, 'any', 'הכול')}`}</span>
                  {rowSentence(item, sentences, locale) && (
                    <span className="constraint-channel">{rowSentence(item, sentences, locale)}</span>
                  )}
                  <Button type="button" variant="text" className="constraint-delete" onClick={() => setDeleteReview(item)} aria-label={t(locale, 'Review deletion of this constraint', 'סקירת מחיקת האילוץ')}>
                    <Trash2 size={14} />
                  </Button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <ConsequenceDialog
        open={Boolean(deleteReview)}
        locale={locale}
        title={t(locale, 'Delete this stored constraint?', 'למחוק את האילוץ השמור?')}
        description={t(locale, 'Review the exact rule and planning effect before it is removed.', 'בדקו את הכלל המדויק ואת השפעתו על התכנון לפני הסרתו.')}
        object={deleteReview ? (
          <span className="consequence-review__object">
            {effectLabel(deleteReview.effect, locale)}
            {rowSentence(deleteReview, sentences, locale) ? ` — ${rowSentence(deleteReview, sentences, locale)}` : ''}
            {' · ID '}<bdi>{String(deleteReview.constraint_id ?? deleteReview.id)}</bdi>
          </span>
        ) : ''}
        scope={deleteReview ? (
          deleteReview.where
            ? t(locale, 'This one stored constraint and its filter predicate. Every other constraint remains unchanged.', 'האילוץ השמור הזה ותנאי הסינון שלו בלבד. כל שאר האילוצים נשארים ללא שינוי.')
            : t(
              locale,
              `This one stored constraint in scope ${deleteReview.scope_type || 'always'}: ${deleteReview.scope_value || 'all values'}${deleteReview.channel ? ` on ${deleteReview.channel}` : ''}. Every other constraint remains unchanged.`,
              `האילוץ השמור הזה בלבד בהיקף ${deleteReview.scope_type || 'תמיד'}: ${deleteReview.scope_value || 'כל הערכים'}${deleteReview.channel ? ` בערוץ ${deleteReview.channel}` : ''}. כל שאר האילוצים נשארים ללא שינוי.`,
            )
        ) : ''}
        consequence={t(locale, 'It stops governing future weekly plan runs. The currently saved plan is not recomputed by this deletion and will need a new run.', 'הוא יפסיק לחול בריצות התכנון השבועיות הבאות. המחיקה אינה מחשבת מחדש את התוכנית השמורה, ויהיה צורך להריץ אותה מחדש.')}
        recovery={t(locale, 'A pre-change snapshot is kept on the Restore changes page.', 'תמונת מצב מלפני השינוי נשמרת בעמוד שחזור שינויים.')}
        confirmLabel={t(locale, 'Delete constraint', 'מחיקת האילוץ')}
        workingLabel={t(locale, 'Deleting constraint', 'מוחק את האילוץ')}
        busy={deleting}
        onCancel={() => setDeleteReview(null)}
        onConfirm={confirmDeleteConstraint}
      />
    </section>
  );
}

export default ConstraintBuilder;
