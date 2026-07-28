import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from '@mui/material';
import { Building2, ChevronLeft, ChevronRight, Info, RefreshCcw, Search } from 'lucide-react';
import {
  agencyTitle,
  filterAgencies,
  isSynthetic,
  normalizeAgencies,
  pageText,
  statusKeys,
  statusMeta,
} from './agencies-helpers';
import AgencyDetailDrawer, { SyntheticChip } from './AgencyDetailDrawer';
import './agency-management.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

// One agency card in the list grid. Click opens the record drawer.
function AgencyCard({ row, locale, onOpen }) {
  const status = statusMeta(row.status, locale);
  const Caret = locale === 'he' ? ChevronLeft : ChevronRight;
  const contact = [row.contact_name, row.contact_role].filter(Boolean).join(' · ');
  return (
    <button type="button" className="amz-card agz-card" onClick={() => onOpen(row.agency_id)}>
      <div className="amz-card-head">
        <div className="amz-card-id-wrap">
          <span className="amz-card-id" dir="auto">{agencyTitle(row)}</span>
          <span className="agz-agency-id" dir="ltr">{row.agency_id}</span>
        </div>
        <Caret size={16} className="amz-card-caret" aria-hidden="true" />
      </div>
      <div className="agz-card-chips">
        <span className={`agz-status-chip ${status.tone}`}>{status.label}</span>
        {row.agency_type && <span className="agz-type-chip" dir="auto">{row.agency_type}</span>}
        {isSynthetic(row) && <SyntheticChip locale={locale} />}
      </div>
      {contact && <span className="agz-card-contact" dir="auto">{contact}</span>}
      {row.notes && <span className="amz-card-notes" dir="auto">{row.notes}</span>}
    </button>
  );
}

// The quiet boundary note: what agency rules touch and what they never touch.
function BoundaryNote({ locale }) {
  return (
    <div className="agz-boundary-note" role="note">
      <Info size={14} aria-hidden="true" />
      <p>
        {pageText(locale, 'Agency rules affect daily spot pricing, reported net revenue, and placement preferences. They do not change the weekly plan, viewer retention, or quarter hour settlement.', 'כללי סוכנות משפיעים על תמחור ספוטים יומי, על הכנסה נטו לדיווח ועל העדפות שיבוץ. הם אינם משנים את התוכנית השבועית, את שימור הצופים או את ההתחשבנות ברבעי שעה.')}
      </p>
    </div>
  );
}

function AgencyManager({ copy, locale, notify, onGlobalRefresh }) {
  const [agencies, setAgencies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState(true);
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState('all');
  const [openId, setOpenId] = useState(null);
  const [scopeOptions, setScopeOptions] = useState({});

  // The scope vocabulary (positions, genres, dayparts, programmes) is shared
  // with the advertiser rules and served by the advertisers options endpoint.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch(`${API_BASE}/api/advertisers/options`);
        if (!response.ok) {
          return;
        }
        const payload = await response.json();
        if (!cancelled) {
          setScopeOptions(payload || {});
        }
      } catch {
        // Options are an enhancement: the chips fall back to local presets.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const loadAgencies = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/agencies`);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      setAgencies(normalizeAgencies(await response.json()));
      setOnline(true);
    } catch {
      setAgencies([]);
      setOnline(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAgencies();
  }, [loadAgencies]);

  const statuses = useMemo(() => statusKeys(agencies), [agencies]);
  const visible = useMemo(
    () => filterAgencies(agencies, { search, status }),
    [agencies, search, status],
  );
  const openRow = useMemo(
    () => agencies.find((row) => row.agency_id === openId) || null,
    [agencies, openId],
  );
  const hasActiveQuery = Boolean(search.trim()) || status !== 'all';

  async function handleSaved() {
    await loadAgencies();
    onGlobalRefresh?.();
  }

  return (
    <section className="page-workspace">
      <div className="page-header">
        <div>
          <h1>{pageText(locale, 'Agencies', 'סוכנויות')}</h1>
          <p>
            {pageText(locale, 'A record per media agency: contacts, commercial terms, the advertisers it books, and its pricing rules. Click a card to open the full record.', 'כרטיס לכל סוכנות מדיה: אנשי קשר, תנאים מסחריים, המפרסמים שהיא מנהלת וכללי התמחור שלה. לחיצה על כרטיס פותחת את הרשומה המלאה.')}
          </p>
        </div>
        <Button className="secondary-button compact" type="button" variant="outlined" onClick={loadAgencies}>
          <RefreshCcw size={14} />
          {copy?.refresh || pageText(locale, 'Refresh', 'רענון')}
        </Button>
      </div>

      <BoundaryNote locale={locale} />

      <div className="amz-toolbar">
        <div className="amz-search">
          <Search size={15} />
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={pageText(locale, 'Search by name, alias, contact or notes', 'חיפוש לפי שם, כינוי, איש קשר או הערות')}
            aria-label={pageText(locale, 'Search agencies', 'חיפוש סוכנויות')}
          />
        </div>
        <div className="amz-filter-chips" role="group" aria-label={pageText(locale, 'Filter by status', 'סינון לפי סטטוס')}>
          <button
            type="button"
            className={`adv-chip${status === 'all' ? ' active' : ''}`}
            aria-pressed={status === 'all'}
            onClick={() => setStatus('all')}
          >
            {pageText(locale, 'All', 'הכול')}
          </button>
          {statuses.map((key) => (
            <button
              key={key}
              type="button"
              className={`adv-chip${status === key ? ' active' : ''}`}
              aria-pressed={status === key}
              onClick={() => setStatus(key)}
            >
              {statusMeta(key, locale).label}
            </button>
          ))}
        </div>
      </div>

      {loading && <div className="amz-empty">{pageText(locale, 'Loading agencies...', 'טוען סוכנויות...')}</div>}

      {!loading && !online && (
        <div className="amz-status-banner" role="alert">
          <Info size={16} aria-hidden="true" />
          <p>
            {pageText(locale, 'The agencies API is unavailable, so no agencies can be shown. This is a connection or deployment gap, not an empty list. Refresh to try again.', 'ה־API של הסוכנויות אינו זמין, ולכן לא ניתן להציג סוכנויות. זהו פער חיבור או פריסה, לא רשימה ריקה. רעננו כדי לנסות שוב.')}
          </p>
        </div>
      )}

      {!loading && online && agencies.length === 0 && (
        <div className="amz-empty">
          <Building2 size={22} />
          {pageText(locale, 'No agencies yet. Records will appear here once agency data is loaded.', 'אין עדיין סוכנויות. רשומות יופיעו כאן ברגע שייטענו נתוני סוכנויות.')}
        </div>
      )}

      {!loading && online && agencies.length > 0 && visible.length === 0 && (
        <div className="amz-empty">
          <span>{pageText(locale, 'No agencies match your search or filter.', 'אין סוכנויות שתואמות את החיפוש או הסינון.')}</span>
          <Button
            className="secondary-button compact"
            type="button"
            variant="outlined"
            onClick={() => {
              setSearch('');
              setStatus('all');
            }}
          >
            {pageText(locale, 'Clear filters', 'ניקוי סינון')}
          </Button>
        </div>
      )}

      {!loading && online && visible.length > 0 && (
        <div className="amz-grid">
          {visible.map((row) => (
            <AgencyCard key={row.agency_id} row={row} locale={locale} onOpen={setOpenId} />
          ))}
        </div>
      )}

      {!loading && online && visible.length > 0 && hasActiveQuery && (
        <div className="amz-result-note">
          {pageText(locale, `Showing ${visible.length} of ${agencies.length}`, `מוצגות ${visible.length} מתוך ${agencies.length}`)}
        </div>
      )}

      <AgencyDetailDrawer
        row={openRow}
        open={Boolean(openRow)}
        locale={locale}
        scopeOptions={scopeOptions}
        notify={notify}
        onSaved={handleSaved}
        onClose={() => setOpenId(null)}
      />
    </section>
  );
}

export default AgencyManager;
