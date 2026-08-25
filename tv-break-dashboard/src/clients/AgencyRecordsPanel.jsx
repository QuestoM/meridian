import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Code, Name } from '../shell/bidi';
import { Button } from '../studio/actions';
import { Building2, ChevronLeft, ChevronRight, Info, RefreshCcw, Search } from 'lucide-react';
import {
  agencyTitle,
  filterAgencies,
  isSynthetic,
  normalizeAgencies,
  normalizeAgencySummary,
  pageText,
  statusKeys,
  statusMeta,
} from './agencies-helpers';
import { formatCurrency, formatNumber } from '../shell/surface-helpers';
import { InputControl } from '../studio/dom-controls';
import AgencyDetailDrawer, { SyntheticChip } from './AgencyDetailDrawer';
import './agency-management.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';
const AGENCY_WINDOW = 18;

// One agency card in the list grid. Click opens the record drawer.
function AgencyCard({ row, locale, onOpen }) {
  const status = statusMeta(row.status, locale);
  const Caret = locale === 'he' ? ChevronLeft : ChevronRight;
  const contact = [row.contact_name, row.contact_role].filter(Boolean).join(' · ');
  return (
    <Button type="button" className="amz-card agz-card" onClick={() => onOpen(row.agency_id)}>
      <div className="amz-card-head">
        <div className="amz-card-id-wrap">
          <Name className="amz-card-id">{agencyTitle(row)}</Name>
          <Code className="agz-agency-id">{row.agency_id}</Code>
        </div>
        <Caret size={16} className="amz-card-caret" aria-hidden="true" />
      </div>
      <div className="agz-card-chips">
        <span className={`agz-status-chip ${status.tone}`}>{status.label}</span>
        {row.agency_type && <Name className="agz-type-chip">{row.agency_type}</Name>}
        {isSynthetic(row) && <SyntheticChip locale={locale} focusable={false} />}
      </div>
      {contact && <Name className="agz-card-contact">{contact}</Name>}
      {row.notes && <Name className="amz-card-notes">{row.notes}</Name>}
    </Button>
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

// Compact totals strip for the daily ledger money. The headline figure is net
// revenue AFTER agency rebates, which is a reporting figure distinct from the
// retention net; the basis (the daily ledger and its source file) is stated on
// the strip so the two nets can never be confused. Honest empty state when no
// daily file is loaded; nothing renders while the summary is still loading.
function LedgerTotalsStrip({ summary, locale, setActiveView }) {
  if (summary === undefined) {
    return null;
  }
  if (!summary || !summary.available) {
    return (
      <div className="agz-totals-strip empty" role="note">
        <span className="agz-subnote">{pageText(locale, 'No daily spot file is loaded, so there are no gross or net totals to show.', 'לא טעון קובץ ספוטים יומי, ולכן אין סכומי ברוטו או נטו להצגה.')}</span>
      </div>
    );
  }
  const basis = summary.basis
    ? pageText(locale, `Basis: the daily ledger (${summary.basis}). This net is the reporting net after agency rebates, not the retention net.`, `הבסיס: הלדג'ר היומי (${summary.basis}). זהו נטו לדיווח אחרי רבייט סוכנויות, לא הנטו של עלות השימור.`)
    : pageText(locale, 'Basis: the daily ledger. This net is the reporting net after agency rebates, not the retention net.', "הבסיס: הלדג'ר היומי. זהו נטו לדיווח אחרי רבייט סוכנויות, לא הנטו של עלות השימור.");
  return (
    <div className="agz-totals-strip" role="group" aria-label={pageText(locale, 'Daily ledger totals', 'סיכומי הלדג\'ר היומי')}>
      <div className="agz-total agz-total-net">
        <span className="agz-total-label">{pageText(locale, 'Net revenue after agency rebates', 'הכנסה נטו אחרי רבייט סוכנויות')}</span>
        <span className="agz-total-value bidi-figure figure-nowrap">{formatCurrency(summary.net, locale)}</span>
      </div>
      <div className="agz-total">
        <span className="agz-total-label">{pageText(locale, 'Gross revenue', 'הכנסה ברוטו')}</span>
        <span className="agz-total-value bidi-figure figure-nowrap">{formatCurrency(summary.gross, locale)}</span>
      </div>
      <div className="agz-total">
        <span className="agz-total-label">{pageText(locale, 'Agency rebates', 'רבייט סוכנויות')}</span>
        <span className="agz-total-value bidi-figure figure-nowrap">{formatCurrency(summary.rebate, locale)}</span>
      </div>
      <div className="agz-total">
        <span className="agz-total-label">{pageText(locale, 'Priced spots', 'ספוטים מתומחרים')}</span>
        <span className="agz-total-value bidi-figure figure-nowrap">{formatNumber(summary.spots, locale)}</span>
      </div>
      <div className="agz-totals-basis">
        <span className="agz-subnote">{basis}</span>
        {typeof setActiveView === 'function' && (
          <Button type="button" className="agz-totals-link" onClick={() => setActiveView('Reports')}>
            {pageText(locale, 'Open the ledger on the Reports page', "לצפייה בלדג'ר בעמוד הדוחות")}
          </Button>
        )}
      </div>
    </div>
  );
}

export function AgencyRecordsPanel({
  copy,
  locale,
  notify,
  onGlobalRefresh,
  setActiveView,
  openAgencyId = '',
  onOpened = () => {},
}) {
  const [agencies, setAgencies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState(true);
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState('all');
  const [openId, setOpenId] = useState(null);
  const [scopeOptions, setScopeOptions] = useState({});
  // undefined = still loading (strip hidden), null = fetch failed (honest empty
  // state), object = normalized summary payload.
  const [summary, setSummary] = useState(undefined);
  const [visibleCount, setVisibleCount] = useState(AGENCY_WINDOW);

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

  const loadSummary = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/agencies/summary`);
      if (!response.ok) {
        throw new Error(String(response.status));
      }
      setSummary(normalizeAgencySummary(await response.json()));
    } catch {
      setSummary(null);
    }
  }, []);

  useEffect(() => {
    loadAgencies();
    loadSummary();
  }, [loadAgencies, loadSummary]);

  // A caller that already resolved an agency opens it here, which is how the
  // agency named on a client record becomes the record itself. The search box is
  // set to the same id so the grid behind the drawer holds that one card, and
  // the request is cleared so closing the drawer does not reopen it.
  useEffect(() => {
    if (!openAgencyId) {
      return;
    }
    setSearch(openAgencyId);
    setStatus('all');
    setOpenId(openAgencyId);
    onOpened();
  }, [openAgencyId, onOpened]);

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
  const windowed = visible.slice(0, visibleCount);

  useEffect(() => {
    setVisibleCount(AGENCY_WINDOW);
  }, [search, status]);

  async function handleSaved() {
    await loadAgencies();
    // A rebate or status edit changes the reported net, so refresh the totals.
    loadSummary();
    onGlobalRefresh?.();
  }

  return (
    <section className="page-workspace" aria-busy={loading} aria-label={pageText(locale, 'Agency records', 'כרטיסי סוכנות')}>
      <div className="page-header">
        <div>
          <h2>{pageText(locale, 'Agencies', 'סוכנויות')}</h2>
          <p>
            {pageText(locale, 'A record per media agency: contacts, commercial terms, the advertisers it books, and its pricing rules. Click a card to open the full record.', 'כרטיס לכל סוכנות מדיה: אנשי קשר, תנאים מסחריים, המפרסמים שהיא מנהלת וכללי התמחור שלה. לחיצה על כרטיס פותחת את הרשומה המלאה.')}
          </p>
        </div>
        <Button
          className="secondary-button compact"
          type="button"
          variant="outlined"
          onClick={() => {
            loadAgencies();
            loadSummary();
          }}
        >
          <RefreshCcw size={14} />
          {copy?.refresh || pageText(locale, 'Refresh', 'רענון')}
        </Button>
      </div>

      <BoundaryNote locale={locale} />

      <LedgerTotalsStrip summary={summary} locale={locale} setActiveView={setActiveView} />

      <div className="amz-toolbar">
        <div className="amz-search">
          <Search size={15} />
          <InputControl
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={pageText(locale, 'Search by name, alias, contact or notes', 'חיפוש לפי שם, כינוי, איש קשר או הערות')}
            aria-label={pageText(locale, 'Search agencies', 'חיפוש סוכנויות')}
          />
        </div>
        <div className="amz-filter-chips" role="group" aria-label={pageText(locale, 'Filter by status', 'סינון לפי סטטוס')}>
          <Button
            type="button"
            className={`adv-chip${status === 'all' ? ' active' : ''}`}
            aria-pressed={status === 'all'}
            onClick={() => setStatus('all')}
          >
            {pageText(locale, 'All', 'הכול')}
          </Button>
          {statuses.map((key) => (
            <Button
              key={key}
              type="button"
              className={`adv-chip${status === key ? ' active' : ''}`}
              aria-pressed={status === key}
              onClick={() => setStatus(key)}
            >
              {statusMeta(key, locale).label}
            </Button>
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
        <div className="amz-grid" aria-label={pageText(locale, 'Agencies', 'סוכנויות')}>
          {windowed.map((row) => (
            <AgencyCard key={row.agency_id} row={row} locale={locale} onOpen={setOpenId} />
          ))}
        </div>
      )}

      {!loading && online && visible.length > 0 && (hasActiveQuery || windowed.length < visible.length) && (
        <div className="amz-result-note clients-window-more" role="status">
          <span>
            {pageText(
              locale,
              `Showing ${windowed.length} of ${visible.length} matching agencies (${agencies.length} total)`,
              `מוצגות ${windowed.length} מתוך ${visible.length} סוכנויות תואמות (${agencies.length} בסך הכול)`,
            )}
          </span>
          {windowed.length < visible.length ? (
            <Button type="button" variant="outlined" className="clients-secondary" onClick={() => setVisibleCount((count) => count + AGENCY_WINDOW)}>
              {pageText(locale, 'Show the next agencies', 'הציגו את הסוכנויות הבאות')}
            </Button>
          ) : null}
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

export default AgencyRecordsPanel;
