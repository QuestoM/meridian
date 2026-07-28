import React, { useCallback, useEffect, useState } from 'react';
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, Drawer, TextField, Tooltip } from '@mui/material';
import { Link2, Power, RotateCcw, Save, X } from 'lucide-react';
import {
  AGENCY_NUMBER_FIELDS,
  isAgencyDirty,
  isSynthetic,
  linkSourceLabel,
  normalizeAgencyConditions,
  normalizeLinks,
  pageText,
  statusMeta,
  toAgencyPayload,
} from './agencies-helpers';
import { toConditionPayload } from './advertisers-helpers';
import AdvertiserConditions from './AdvertiserConditions';
import './agency-management.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

// The record-level synthetic marker. Bottom-placed tooltip (never a native
// title) explains that the details are seed data pending real operator data.
export function SyntheticChip({ locale }) {
  return (
    <Tooltip
      title={pageText(locale, 'The details on this record are synthetic seed data, generated to stand the page up until real agency data arrives from the operator. Do not treat them as facts.', 'הפרטים ברשומה זו הם נתוני דמו סינתטיים שנוצרו כדי להקים את העמוד, עד שיתקבלו נתוני סוכנות אמיתיים מהמפעיל. אין להתייחס אליהם כאל עובדות.')}
      arrow
      placement="bottom"
    >
      <span className="agz-synthetic-chip" tabIndex={0}>{pageText(locale, 'Demo data', 'נתוני דמו')}</span>
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
        inputProps={{ 'aria-label': label, ...(ltr ? { dir: 'ltr' } : {}) }}
      />
    </div>
  );
}

// One contact block (primary or secondary): name, role, phone, email.
function ContactFields({ prefix, title, draft, update, locale }) {
  const key = (name) => (prefix ? `${prefix}_${name}` : name);
  return (
    <fieldset className="agz-contact-block">
      <legend>{title}</legend>
      <div className="agz-field-grid">
        <Field label={pageText(locale, 'Name', 'שם')} value={draft[key('contact_name')]} onChange={(value) => update(key('contact_name'), value)} />
        <Field label={pageText(locale, 'Role', 'תפקיד')} value={draft[key('contact_role')]} onChange={(value) => update(key('contact_role'), value)} />
        <Field label={pageText(locale, 'Phone', 'טלפון')} value={draft[key('contact_phone')]} onChange={(value) => update(key('contact_phone'), value)} ltr />
        <Field label={pageText(locale, 'Email', 'דוא״ל')} value={draft[key('contact_email')]} onChange={(value) => update(key('contact_email'), value)} ltr />
      </div>
    </fieldset>
  );
}

// The linked-advertisers section: each link carries its provenance (observed
// in the spot data vs manually linked by an operator).
function LinkedAdvertisers({ state, locale }) {
  return (
    <section className="amz-drawer-section">
      <h3>{pageText(locale, 'Linked advertisers', 'מפרסמים מקושרים')}</h3>
      {state.status === 'loading' && (
        <p className="agz-subnote">{pageText(locale, 'Loading links...', 'טוען קישורים...')}</p>
      )}
      {state.status === 'error' && (
        <p className="agz-inline-warn" role="note">{pageText(locale, 'Advertiser links could not be loaded. This is a load failure, not an empty list.', 'קישורי המפרסמים לא נטענו. זהו כשל טעינה, לא רשימה ריקה.')}</p>
      )}
      {state.status === 'ready' && state.links.length === 0 && (
        <p className="agz-subnote">{pageText(locale, 'No advertisers are linked to this agency yet.', 'אין עדיין מפרסמים המקושרים לסוכנות זו.')}</p>
      )}
      {state.status === 'ready' && state.links.length > 0 && (
        <ul className="agz-link-list">
          {state.links.map((link) => (
            <li key={`${link.advertiser}-${link.source}`} className="agz-link-row">
              <Link2 size={13} aria-hidden="true" />
              <span className="agz-link-name" dir="ltr">{link.advertiser}</span>
              <span className={`agz-status-chip ${link.source === 'manual' ? 'blue' : 'teal'}`}>{linkSourceLabel(link.source, locale)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

// Full record editor + linked advertisers + conditions builder for one agency.
function AgencyDetailDrawer({ row, open, locale, scopeOptions, notify, onSaved, onClose }) {
  const [draft, setDraft] = useState(row);
  const [saving, setSaving] = useState(false);
  const [confirmDeactivate, setConfirmDeactivate] = useState(false);
  const [linksState, setLinksState] = useState({ status: 'loading', links: [] });
  const [condState, setCondState] = useState({ status: 'loading', conditions: [] });

  const agencyId = row ? row.agency_id : null;

  useEffect(() => {
    setDraft(row);
    setConfirmDeactivate(false);
  }, [row]);

  const loadLinks = useCallback(async () => {
    if (!agencyId) {
      return;
    }
    setLinksState({ status: 'loading', links: [] });
    try {
      const response = await fetch(`${API_BASE}/api/agencies/${encodeURIComponent(agencyId)}/advertisers`);
      if (!response.ok) {
        throw new Error(String(response.status));
      }
      setLinksState({ status: 'ready', links: normalizeLinks(await response.json()) });
    } catch {
      setLinksState({ status: 'error', links: [] });
    }
  }, [agencyId]);

  const loadConditions = useCallback(async () => {
    if (!agencyId) {
      return;
    }
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
    if (open && agencyId) {
      loadLinks();
      loadConditions();
    }
  }, [open, agencyId, loadLinks, loadConditions]);

  if (!row) {
    return null;
  }

  const dirty = isAgencyDirty(row, draft);
  const status = statusMeta(draft.status, locale);
  const active = status.key !== 'inactive';
  const anchor = locale === 'he' ? 'left' : 'right';
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
      notify(`Agency ${row.agency_id} saved.`, `הסוכנות ${row.agency_id} נשמרה.`);
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
      const en = nextStatus === 'inactive' ? `Agency ${row.agency_id} deactivated.` : `Agency ${row.agency_id} reactivated.`;
      const he = nextStatus === 'inactive' ? `הסוכנות ${row.agency_id} הושבתה.` : `הסוכנות ${row.agency_id} הופעלה מחדש.`;
      notify(en, he);
      await onSaved();
    } catch (error) {
      notify(`Status change failed for ${row.agency_id} (${error.message}).`, `שינוי הסטטוס נכשל עבור ${row.agency_id} (${error.message}).`);
    } finally {
      setConfirmDeactivate(false);
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
    <Drawer
      anchor={anchor}
      open={open}
      onClose={onClose}
      slotProps={{ paper: { className: 'amz-drawer-paper', dir: locale === 'he' ? 'rtl' : 'ltr' } }}
    >
      <div className="amz-drawer">
        <header className="amz-drawer-head">
          <div className="amz-drawer-title">
            <span className="amz-drawer-eyebrow">{pageText(locale, 'Agency record', 'כרטיס סוכנות')}</span>
            <h2 dir="auto">{draft.display_name || draft.name || draft.agency_id}</h2>
            <div className="agz-head-chips">
              <span className="agz-agency-id" dir="ltr">{row.agency_id}</span>
              <span className={`agz-status-chip ${status.tone}`}>{status.label}</span>
              {isSynthetic(row) && <SyntheticChip locale={locale} />}
              {row.onboarded_at && (
                <span className="agz-subnote" dir="auto">{pageText(locale, `Onboarded ${row.onboarded_at}`, `הצטרפה ${row.onboarded_at}`)}</span>
              )}
            </div>
          </div>
          <button type="button" className="amz-drawer-close" onClick={onClose} aria-label={pageText(locale, 'Close', 'סגירה')}>
            <X size={18} />
          </button>
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
          <ContactFields prefix="" title={pageText(locale, 'Primary contact', 'איש קשר ראשי')} draft={draft} update={update} locale={locale} />
          <ContactFields prefix="secondary" title={pageText(locale, 'Secondary contact', 'איש קשר משני')} draft={draft} update={update} locale={locale} />
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
              overlaps={[]}
              locale={locale}
              scopeOptions={scopeOptions}
              onCreate={handleCreateCondition}
              onUpdate={handleUpdateCondition}
              onDelete={handleDeleteCondition}
            />
          )}
        </section>

        <footer className="amz-drawer-foot">
          {active ? (
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setConfirmDeactivate(true)}>
              <Power size={14} />
              {pageText(locale, 'Deactivate agency', 'השבתת הסוכנות')}
            </Button>
          ) : (
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setStatusTo('active')}>
              <Power size={14} />
              {pageText(locale, 'Reactivate agency', 'הפעלה מחדש של הסוכנות')}
            </Button>
          )}
        </footer>
      </div>

      <Dialog open={confirmDeactivate} onClose={() => setConfirmDeactivate(false)} dir={locale === 'he' ? 'rtl' : 'ltr'}>
        <DialogTitle>{pageText(locale, 'Deactivate this agency', 'השבתת הסוכנות')}</DialogTitle>
        <DialogContent>
          <p className="agz-dialog-text">
            {pageText(locale, `The agency ${row.agency_id} will be marked inactive. Its advertiser links and rules are kept, nothing is deleted, and it can be reactivated at any time.`, `הסוכנות ${row.agency_id} תסומן כלא פעילה. קישורי המפרסמים והכללים שלה נשמרים, דבר אינו נמחק, וניתן להפעיל אותה מחדש בכל עת.`)}
          </p>
        </DialogContent>
        <DialogActions>
          <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setConfirmDeactivate(false)}>
            {pageText(locale, 'Cancel', 'ביטול')}
          </Button>
          <Button className="secondary-button compact danger" type="button" variant="outlined" onClick={() => setStatusTo('inactive')}>
            {pageText(locale, 'Deactivate', 'השבתה')}
          </Button>
        </DialogActions>
      </Dialog>
    </Drawer>
  );
}

export default AgencyDetailDrawer;
