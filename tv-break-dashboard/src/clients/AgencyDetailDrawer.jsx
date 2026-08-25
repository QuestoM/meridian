import React, { useCallback, useEffect, useState } from 'react';
import { Code, Name } from '../shell/bidi';
import { Drawer, TextField, Tooltip } from '@mui/material';
import { Dialog } from '../studio/modal';
import { Button } from '../studio/actions';
import { Power, RotateCcw, Save, X } from 'lucide-react';
import {
  isAgencyDirty,
  isSynthetic,
  linksSourceFile,
  normalizeAgencyConditions,
  normalizeLinks,
  pageText,
  statusMeta,
  toAgencyPayload,
} from './agencies-helpers';
import { toConditionPayload } from './advertisers-helpers';
import { useAssistantEntity } from '../shell/assistant-page-context';
import AdvertiserConditions from './AdvertiserConditions';
import LinkedAdvertisers from './AgencyLinkedAdvertisers';
import './agency-management.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

// The record-level synthetic marker. Bottom-placed tooltip (never a native
// title) explains that the details are seed data pending real operator data.
export function SyntheticChip({ locale, focusable = true }) {
  const explanation = pageText(locale, 'The details on this record are synthetic seed data, generated to stand the page up until real agency data arrives from the operator. Do not treat them as facts.', 'הפרטים ברשומה זו הם נתוני דמו סינתטיים שנוצרו כדי להקים את העמוד, עד שיתקבלו נתוני סוכנות אמיתיים מהמפעיל. אין להתייחס אליהם כאל עובדות.');
  return (
    <Tooltip
      title={explanation}
      arrow
      placement="bottom"
    >
      <span
        className="agz-synthetic-chip"
        tabIndex={focusable ? 0 : undefined}
        role={focusable ? 'note' : undefined}
        aria-label={focusable ? explanation : undefined}
      >
        {pageText(locale, 'Demo data', 'נתוני דמו')}
      </span>
    </Tooltip>
  );
}

// One labeled text input bound to a draft field.
function Field({ label, value, onChange, type = 'text', ltr = false, full = false }) {
  return (
    <div className={`amz-drawer-field${full ? ' agz-field-full' : ''}`}>
      <span className="adv-field-label">{label}</span>
      <TextField
        size="small"
        type={type}
        value={value ?? ''}
        onChange={(event) => onChange(event.target.value)}
        slotProps={{ htmlInput: { 'aria-label': label, ...(ltr ? { dir: 'ltr' } : {}) } }}
      />
    </div>
  );
}

// One contact block bound to the API's field names: prefix "contact" edits the
// primary contact_* columns, prefix "contact2" the secondary contact2_* ones.
function ContactFields({ prefix, title, draft, update, locale }) {
  const key = (name) => `${prefix}_${name}`;
  return (
    <fieldset className="agz-contact-block">
      <legend>{title}</legend>
      <div className="agz-field-grid">
        <Field label={pageText(locale, 'Name', 'שם')} value={draft[key('name')]} onChange={(value) => update(key('name'), value)} />
        <Field label={pageText(locale, 'Role', 'תפקיד')} value={draft[key('role')]} onChange={(value) => update(key('role'), value)} />
        <Field label={pageText(locale, 'Phone', 'טלפון')} value={draft[key('phone')]} onChange={(value) => update(key('phone'), value)} ltr />
        <Field label={pageText(locale, 'Email', 'דוא״ל')} value={draft[key('email')]} onChange={(value) => update(key('email'), value)} ltr />
      </div>
    </fieldset>
  );
}

// The drawer body for ONE agency record. Mounted only when a row exists and
// keyed by agency_id, so the draft state initializes from a real record: the
// old always-mounted variant read draft.status while draft was still null on
// the first open render, which crashed the whole page white.
function AgencyDrawerBody({ row, locale, scopeOptions, notify, onSaved, onClose }) {
  const [draft, setDraft] = useState(row);
  const [saving, setSaving] = useState(false);
  const [confirmSuspend, setConfirmSuspend] = useState(false);
  const [linksState, setLinksState] = useState({ status: 'loading', links: [], sourceFile: null });
  const [condState, setCondState] = useState({ status: 'loading', conditions: [] });

  const agencyId = row.agency_id;

  useAssistantEntity('agency', agencyId, row.display_name || row.name || agencyId);

  // After a save reloads the list, the row object is replaced; follow it.
  useEffect(() => {
    setDraft(row);
  }, [row]);

  const loadLinks = useCallback(async () => {
    setLinksState({ status: 'loading', links: [], sourceFile: null });
    try {
      const response = await fetch(`${API_BASE}/api/agencies/${encodeURIComponent(agencyId)}/advertisers`);
      if (!response.ok) {
        throw new Error(String(response.status));
      }
      const payload = await response.json();
      setLinksState({
        status: 'ready',
        links: normalizeLinks(payload),
        sourceFile: linksSourceFile(payload),
      });
    } catch {
      setLinksState({ status: 'error', links: [], sourceFile: null });
    }
  }, [agencyId]);

  const loadConditions = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/agencies/${encodeURIComponent(agencyId)}/conditions`);
      if (!response.ok) {
        throw new Error(String(response.status));
      }
      setCondState({ status: 'ready', conditions: normalizeAgencyConditions(await response.json()) });
    } catch {
      setCondState({ status: 'error', conditions: [] });
    }
  }, [agencyId]);

  useEffect(() => {
    loadLinks();
    loadConditions();
  }, [loadLinks, loadConditions]);

  const dirty = isAgencyDirty(row, draft);
  const status = statusMeta(draft.status, locale);
  // The API status vocabulary is exactly active | suspended: the footer keys
  // off suspended so a suspended agency always shows Reactivate.
  const suspended = status.key === 'suspended';
  const update = (field, value) => setDraft((current) => ({ ...current, [field]: value }));

  async function putAgency(body) {
    const response = await fetch(`${API_BASE}/api/agencies/${encodeURIComponent(row.agency_id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
  }

  async function handleSave() {
    setSaving(true);
    try {
      await putAgency(toAgencyPayload(draft));
      notify(`Agency ${row.name || row.agency_id} saved.`, `הסוכנות ${row.name || row.agency_id} נשמרה.`);
      await onSaved();
    } catch (error) {
      notify(`Save failed for ${row.agency_id} (${error.message}).`, `השמירה נכשלה עבור ${row.agency_id} (${error.message}).`);
    } finally {
      setSaving(false);
    }
  }

  async function setStatusTo(nextStatus) {
    try {
      await putAgency(toAgencyPayload({ ...draft, status: nextStatus }));
      const en = nextStatus === 'suspended' ? `Agency ${row.agency_id} suspended.` : `Agency ${row.agency_id} reactivated.`;
      const he = nextStatus === 'suspended' ? `הסוכנות ${row.agency_id} הושהתה.` : `הסוכנות ${row.agency_id} הופעלה מחדש.`;
      notify(en, he);
      await onSaved();
    } catch (error) {
      notify(`Status change failed for ${row.agency_id} (${error.message}).`, `שינוי הסטטוס נכשל עבור ${row.agency_id} (${error.message}).`);
    } finally {
      setConfirmSuspend(false);
    }
  }

  async function conditionRequest(method, path, body) {
    const response = await fetch(`${API_BASE}/api/agencies/${encodeURIComponent(row.agency_id)}${path}`, {
      method,
      ...(body ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}),
    });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
  }

  async function handleCreateCondition(_agencyId, conditionDraft) {
    try {
      const ruleId = `rule_${Date.now().toString(36)}`;
      await conditionRequest('POST', '/conditions', { rule_id: ruleId, ...toConditionPayload(conditionDraft) });
      notify(`Rule added to agency ${row.agency_id}.`, `כלל נוסף לסוכנות ${row.agency_id}.`);
      await loadConditions();
      return true;
    } catch (error) {
      notify(`Add rule failed (${error.message}).`, `הוספת הכלל נכשלה (${error.message}).`);
      return false;
    }
  }

  async function handleUpdateCondition(_agencyId, ruleId, conditionDraft) {
    try {
      await conditionRequest('PUT', `/conditions/${encodeURIComponent(ruleId)}`, toConditionPayload(conditionDraft));
      notify(`Rule saved for agency ${row.agency_id}.`, `כלל נשמר עבור הסוכנות ${row.agency_id}.`);
      await loadConditions();
      return true;
    } catch (error) {
      notify(`Save rule failed (${error.message}).`, `שמירת הכלל נכשלה (${error.message}).`);
      return false;
    }
  }

  async function handleDeleteCondition(_agencyId, ruleId) {
    try {
      await conditionRequest('DELETE', `/conditions/${encodeURIComponent(ruleId)}`);
      notify(`Rule removed from agency ${row.agency_id}.`, `כלל הוסר מהסוכנות ${row.agency_id}.`);
      await loadConditions();
    } catch (error) {
      notify(`Delete rule failed (${error.message}).`, `מחיקת הכלל נכשלה (${error.message}).`);
    }
  }

  return (
    <>
      <div className="amz-drawer">
        <header className="amz-drawer-head">
          <div className="amz-drawer-title">
            <span className="amz-drawer-eyebrow">{pageText(locale, 'Agency record', 'כרטיס סוכנות')}</span>
            <h2 id="agency-drawer-title"><Name>{draft.display_name || draft.name || draft.agency_id}</Name></h2>
            <div className="agz-head-chips">
              <Code className="agz-agency-id">{row.agency_id}</Code>
              <span className={`agz-status-chip ${status.tone}`}>{status.label}</span>
              {isSynthetic(row) && <SyntheticChip locale={locale} />}
              {row.onboarded_at && (
                <span className="agz-subnote">{pageText(locale, `Onboarded ${row.onboarded_at}`, `הצטרפה ${row.onboarded_at}`)}</span>
              )}
            </div>
          </div>
          <Button autoFocus type="button" className="amz-drawer-close" onClick={onClose} aria-label={pageText(locale, 'Close agency record', 'סגירת כרטיס הסוכנות')}>
            <X size={18} />
          </Button>
        </header>

        <section className="amz-drawer-section">
          <h3>{pageText(locale, 'Agency details', 'פרטי הסוכנות')}</h3>
          <div className="agz-field-grid">
            <Field label={pageText(locale, 'Name', 'שם')} value={draft.name} onChange={(value) => update('name', value)} />
            <Field label={pageText(locale, 'Display name', 'שם תצוגה')} value={draft.display_name} onChange={(value) => update('display_name', value)} />
            <Field label={pageText(locale, 'Aliases (comma separated)', 'כינויים (מופרדים בפסיק)')} value={draft.aliases} onChange={(value) => update('aliases', value)} />
            <Field label={pageText(locale, 'Agency type', 'סוג הסוכנות')} value={draft.agency_type} onChange={(value) => update('agency_type', value)} />
            <Field label={pageText(locale, 'VAT id', 'ח״פ / עוסק')} value={draft.vat_id} onChange={(value) => update('vat_id', value)} ltr />
            <Field label={pageText(locale, 'City', 'עיר')} value={draft.address_city} onChange={(value) => update('address_city', value)} />
            <Field label={pageText(locale, 'Street', 'רחוב')} value={draft.address_street} onChange={(value) => update('address_street', value)} full />
          </div>
        </section>

        <section className="amz-drawer-section">
          <h3>{pageText(locale, 'Contacts', 'אנשי קשר')}</h3>
          <ContactFields prefix="contact" title={pageText(locale, 'Primary contact', 'איש קשר ראשי')} draft={draft} update={update} locale={locale} />
          <ContactFields prefix="contact2" title={pageText(locale, 'Secondary contact', 'איש קשר משני')} draft={draft} update={update} locale={locale} />
        </section>

        <section className="amz-drawer-section">
          <h3>{pageText(locale, 'Commercial terms', 'תנאים מסחריים')}</h3>
          <div className="agz-field-grid">
            <Field label={pageText(locale, 'Payment terms (days)', 'תנאי תשלום (ימים)')} value={draft.payment_terms_days} onChange={(value) => update('payment_terms_days', value)} type="number" ltr />
            <Field label={pageText(locale, 'Rebate (%)', 'רבייט (%)')} value={draft.rebate_percent} onChange={(value) => update('rebate_percent', value)} type="number" ltr />
            <Field label={pageText(locale, 'Commission (%)', 'עמלה (%)')} value={draft.commission_percent} onChange={(value) => update('commission_percent', value)} type="number" ltr />
            <Field label={pageText(locale, 'Credit limit (ILS)', 'מסגרת אשראי (ש״ח)')} value={draft.credit_limit_ils} onChange={(value) => update('credit_limit_ils', value)} type="number" ltr />
            <Field label={pageText(locale, 'Notes', 'הערות')} value={draft.notes} onChange={(value) => update('notes', value)} full />
          </div>
          <div className="amz-baseline-actions">
            <Button className="run-button compact" type="button" variant="contained" disabled={!dirty || saving} onClick={handleSave}>
              <Save size={14} />
              {saving ? pageText(locale, 'Saving...', 'שומר...') : pageText(locale, 'Save agency', 'שמירת הסוכנות')}
            </Button>
            {dirty && (
              <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setDraft(row)}>
                <RotateCcw size={14} />
                {pageText(locale, 'Revert', 'שחזור')}
              </Button>
            )}
          </div>
        </section>

        <LinkedAdvertisers state={linksState} locale={locale} />

        <section className="amz-drawer-section">
          <h3>{pageText(locale, 'Agency rules', 'כללי הסוכנות')}</h3>
          {condState.status === 'error' ? (
            <p className="agz-inline-warn" role="note">{pageText(locale, 'Agency rules could not be loaded. This is a load failure, not an empty rule set.', 'כללי הסוכנות לא נטענו. זהו כשל טעינה, לא היעדר כללים.')}</p>
          ) : (
            <AdvertiserConditions
              advertiserId={row.agency_id}
              conditions={condState.conditions}
              overlaps={row.overlaps}
              locale={locale}
              scopeOptions={scopeOptions}
              onCreate={handleCreateCondition}
              onUpdate={handleUpdateCondition}
              onDelete={handleDeleteCondition}
            />
          )}
        </section>

        <footer className="amz-drawer-foot">
          {suspended ? (
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setStatusTo('active')}>
              <Power size={14} />
              {pageText(locale, 'Reactivate agency', 'הפעלה מחדש של הסוכנות')}
            </Button>
          ) : (
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setConfirmSuspend(true)}>
              <Power size={14} />
              {pageText(locale, 'Suspend agency', 'השהיית הסוכנות')}
            </Button>
          )}
        </footer>
      </div>

      <Dialog
        open={confirmSuspend}
        onClose={() => setConfirmSuspend(false)}
        size="narrow"
        title={pageText(locale, 'Suspend this agency', 'השהיית הסוכנות')}
        closeLabel={pageText(locale, 'Cancel', 'ביטול')}
        footer={(
          <>
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setConfirmSuspend(false)}>
              {pageText(locale, 'Cancel', 'ביטול')}
            </Button>
            <Button className="secondary-button compact danger" type="button" variant="outlined" onClick={() => setStatusTo('suspended')}>
              {pageText(locale, 'Suspend', 'השהיה')}
            </Button>
          </>
        )}
      >
        <p className="agz-dialog-text">
          {pageText(locale, `The agency ${row.agency_id} will be marked suspended. Its rules and rebate go inert on the pricing path, its advertiser links and history are kept, nothing is deleted, and it can be reactivated at any time.`, `הסוכנות ${row.agency_id} תסומן כמושהית. הכללים והרבייט שלה מפסיקים לפעול בנתיב התמחור, קישורי המפרסמים וההיסטוריה נשמרים, דבר אינו נמחק, וניתן להפעיל אותה מחדש בכל עת.`)}
        </p>
      </Dialog>
    </>
  );
}

// Full record editor + linked advertisers + conditions builder for one agency.
// The Drawer shell always mounts (so the open/close transition works); the body
// mounts only with a real row, keyed by agency_id so all per-agency state
// (draft, links, conditions, confirm dialog) resets cleanly between records.
function AgencyDetailDrawer({ row, open, locale, scopeOptions, notify, onSaved, onClose }) {
  const anchor = locale === 'he' ? 'left' : 'right';
  return (
    <Drawer
      anchor={anchor}
      open={open && Boolean(row)}
      onClose={onClose}
      slotProps={{ paper: {
        className: 'amz-drawer-paper',
        dir: locale === 'he' ? 'rtl' : 'ltr',
        role: 'dialog',
        'aria-modal': 'true',
        'aria-labelledby': 'agency-drawer-title',
      } }}
    >
      {row && (
        <AgencyDrawerBody
          key={row.agency_id}
          row={row}
          locale={locale}
          scopeOptions={scopeOptions}
          notify={notify}
          onSaved={onSaved}
          onClose={onClose}
        />
      )}
    </Drawer>
  );
}

export default AgencyDetailDrawer;
