import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Figure } from '../shell/bidi';
import { Button } from '../studio/actions';
import { Info, Plus, RefreshCcw, Search, Users, X } from 'lucide-react';
import { normalizeRows, pageText, suggestNextId } from './advertisers-helpers';
import {
  filterManaged,
  indexIdentityByRow,
  indexStats,
  managementSummary,
  mergeRowWithIdentity,
  mergeRowWithStats,
  sortManaged,
} from './advertiser-stats-helpers';
import { recordWrites } from './advertiser-record-writes';
import { loadAdvertiserIdentity } from './clients-api';
import { InputControl, SelectControl } from '../studio/dom-controls';
import AdvertiserCardGrid from './AdvertiserCardGrid';
import AdvertiserDetailDrawer from './AdvertiserDetailDrawer';
import AddAdvertiserForm from './AddAdvertiserForm';
import './advertiser-management.css';
import './advertiser-names.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';
const ADVERTISER_WINDOW = 18;

// The store this panel edits holds pricing rows, not advertisers. A row prices
// an advertiser only once its name cell carries that advertiser's name, which is
// why an unbound row shows no money: it has none, and the identity read below is
// what turns a bound row's money into the real figure from the daily ledger.

const FILTERS = [
  { key: 'all', en: 'All', he: 'הכול' },
  { key: 'with-rules', en: 'With scoped rules', he: 'עם כללים ממוקדים' },
  { key: 'custom-premium', en: 'Custom premium', he: 'מקדם מותאם' },
  { key: 'conflicts', en: 'Has conflicts', he: 'עם התנגשויות' },
];

export function AdvertiserRecordsPanel({
  copy,
  locale,
  notify,
  onGlobalRefresh,
  openAdvertiserId = '',
  onOpened = () => {},
}) {
  const [advertisers, setAdvertisers] = useState([]);
  const [statsIndex, setStatsIndex] = useState(() => new Map());
  const [identityIndex, setIdentityIndex] = useState(() => new Map());
  const [statusNote, setStatusNote] = useState('');
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState(true);
  const [statsError, setStatsError] = useState(false);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');
  const [sortKey, setSortKey] = useState('name');
  const [showAdd, setShowAdd] = useState(false);
  const [scopeOptions, setScopeOptions] = useState({});
  const [openId, setOpenId] = useState(null);
  const [visibleCount, setVisibleCount] = useState(ADVERTISER_WINDOW);
  // Two-step delete interlock: the first confirmed click arms the id and warns
  // that the advertiser's scoped rules die with it; only the second click deletes.
  const [pendingDeleteId, setPendingDeleteId] = useState(null);

  useEffect(() => {
    setPendingDeleteId(null);
  }, [openId]);

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

  const loadStats = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/advertisers/stats`);
      if (!response.ok) {
        setStatsError(true);
        return;
      }
      const payload = await response.json();
      setStatsIndex(indexStats(payload));
      setStatusNote(payload && payload.status ? String(payload.status) : '');
      setStatsError(false);
    } catch {
      // Stats are an enhancement, but a failure must read as an error, not as
      // empty data: flag it so the cards' "-" carries an honest reason.
      setStatsError(true);
    }
  }, []);

  // Who each bound row prices and what that advertiser actually earned. The read
  // already joins the name space, the rules store and the priced daily ledger,
  // so this panel performs no money arithmetic of its own.
  const loadIdentity = useCallback(async () => {
    try {
      setIdentityIndex(indexIdentityByRow(await loadAdvertiserIdentity()));
    } catch {
      // A failed identity read leaves every figure at its honest dash, which is
      // the same state an unbound row is already in.
      setIdentityIndex(new Map());
    }
  }, []);

  const loadAdvertisers = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/advertisers`);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      const payload = await response.json();
      setAdvertisers(normalizeRows(payload.advertisers));
      setOnline(true);
      await Promise.all([loadStats(), loadIdentity()]);
    } catch {
      setAdvertisers([]);
      setOnline(false);
    } finally {
      setLoading(false);
    }
  }, [loadStats, loadIdentity]);

  useEffect(() => {
    loadAdvertisers();
  }, [loadAdvertisers]);

  // A caller that already resolved a row opens it here. The search box is set to
  // the same id so the grid behind the drawer holds that one card and nothing
  // else, and the request is cleared so closing the drawer does not reopen it.
  useEffect(() => {
    if (!openAdvertiserId) {
      return;
    }
    setSearch(openAdvertiserId);
    setFilter('all');
    setOpenId(openAdvertiserId);
    onOpened();
  }, [openAdvertiserId, onOpened]);

  const merged = useMemo(
    () => advertisers.map((row) => mergeRowWithIdentity(mergeRowWithStats(row, statsIndex), identityIndex)),
    [advertisers, statsIndex, identityIndex],
  );
  const summary = useMemo(() => managementSummary(merged), [merged]);
  const existingIds = useMemo(() => advertisers.map((row) => row.advertiser_id), [advertisers]);
  const suggestedId = useMemo(() => suggestNextId(advertisers), [advertisers]);
  const visible = useMemo(
    () => sortManaged(filterManaged(merged, { search, filter }), sortKey, locale),
    [merged, search, filter, sortKey, locale],
  );
  const openRow = useMemo(
    () => merged.find((row) => row.advertiser_id === openId) || null,
    [merged, openId],
  );
  const hasActiveQuery = Boolean(search.trim()) || filter !== 'all';
  const windowed = visible.slice(0, visibleCount);

  useEffect(() => {
    setVisibleCount(ADVERTISER_WINDOW);
  }, [search, filter, sortKey]);

  const writes = useMemo(
    () => recordWrites({
      base: API_BASE,
      notify,
      reload: loadAdvertisers,
      refreshGlobal: onGlobalRefresh,
      rowsRef: () => merged,
      pending: pendingDeleteId,
      setPending: setPendingDeleteId,
      setOpenId,
      setShowAdd,
    }),
    [notify, loadAdvertisers, onGlobalRefresh, merged, pendingDeleteId],
  );

  function clearQuery() {
    setSearch('');
    setFilter('all');
  }

  const cards = [
    { key: 'total', value: summary.total, en: 'Pricing rows', he: 'שורות תמחור' },
    { key: 'bound', value: summary.bound, en: 'Priced an advertiser', he: 'מתמחרות מפרסם' },
    { key: 'withRules', value: summary.withRules, en: 'With scoped rules', he: 'עם כללים ממוקדים' },
    { key: 'totalRules', value: summary.totalRules, en: 'Scoped rules total', he: 'סך כללים ממוקדים' },
    { key: 'conflicts', value: summary.conflicts, en: 'Conflicts flagged', he: 'התנגשויות שסומנו', warn: true },
  ];

  return (
    <section className="page-workspace" aria-busy={loading} aria-label={pageText(locale, 'Advertiser pricing records', 'רשומות תמחור מפרסמים')}>
      <div className="page-header">
        <div>
          <h2>{pageText(locale, 'Pricing rules', 'כללי תמחור')}</h2>
          <p>
            {pageText(
              locale,
              'One row per pricing rule: which advertiser it prices, its premium, its scoped rules and its conflicts. A row that carries no advertiser name prices nobody. A client gets its rule from its own record, under Clients.',
              'שורה אחת לכל כלל תמחור: איזה מפרסם היא מתמחרת, המקדם שלה, הכללים הממוקדים שלה וההתנגשויות. שורה שאינה נושאת שם מפרסם אינה מתמחרת אף אחד. לקוח מקבל את הכלל שלו מהכרטיס שלו, במסך הלקוחות.',
            )}
          </p>
        </div>
        <Button className="secondary-button compact" type="button" variant="outlined" onClick={loadAdvertisers}>
          <RefreshCcw size={14} />
          {copy?.refresh || pageText(locale, 'Refresh', 'רענון')}
        </Button>
      </div>

      {!loading && online && advertisers.length > 0 && (
        <div className="amz-summary">
          {cards.map((card) => (
            <div className={`amz-summary-card${card.warn && card.value > 0 ? ' warn' : ''}`} key={card.key}>
              <Figure className="amz-summary-value">{card.value}</Figure>
              <span className="amz-summary-label">{pageText(locale, card.en, card.he)}</span>
            </div>
          ))}
        </div>
      )}

      {statsError && !loading && online && (
        <div className="amz-status-banner" role="alert">
          <Info size={16} aria-hidden="true" />
          <p>
            {pageText(
              locale,
              'Stats did not load. The figures below show "-" because the stats request failed, not because the data is empty. Refresh to try again.',
              'הנתונים הסטטיסטיים לא נטענו. הערכים מטה מוצגים כ-״-״ כי בקשת הנתונים נכשלה, לא כי אין נתונים. רעננו כדי לנסות שוב.',
            )}
          </p>
        </div>
      )}

      {statusNote && !loading && online && (
        <div className="amz-status-banner" role="note">
          <Info size={16} aria-hidden="true" />
          <p>
            {pageText(
              locale,
              `Honest status: ${statusNote} Revenue below is the priced daily ledger's own figure for the advertiser a row prices, so a row that prices nobody shows "-", and profitability stays "-" because cost per advertiser is not exposed anywhere.`,
              'מצב שקוף: המנוע השבועי אינו צורך את כללי המפרסמים; רק מסלול תמחור הספוטים היומי מתמחר מולם. ההכנסה מטה היא הסכום של הפנקס היומי המתומחר עבור המפרסם שהשורה מתמחרת, ולכן שורה שאינה מתמחרת אף אחד מציגה ״-״, והרווחיות נשארת ״-״ משום שעלות לכל מפרסם אינה חשופה בשום מקום.',
            )}
          </p>
        </div>
      )}

      <div className="amz-toolbar">
        <div className="amz-search">
          <Search size={15} />
          <InputControl
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={pageText(locale, 'Search by name, ID or notes', 'חיפוש לפי שם, מזהה או הערות')}
            aria-label={pageText(locale, 'Search advertisers', 'חיפוש מפרסמים')}
          />
        </div>
        <div className="amz-filter-chips" role="group" aria-label={pageText(locale, 'Filter advertisers', 'סינון מפרסמים')}>
          {FILTERS.map((entry) => (
            <Button
              key={entry.key}
              type="button"
              className={`adv-chip${filter === entry.key ? ' active' : ''}`}
              aria-pressed={filter === entry.key}
              onClick={() => setFilter(entry.key)}
            >
              {pageText(locale, entry.en, entry.he)}
            </Button>
          ))}
        </div>
        <div className="amz-sort">
          <label htmlFor="amz-sort-select">{pageText(locale, 'Sort', 'מיון')}</label>
          <SelectControl id="amz-sort-select" value={sortKey} onChange={(event) => setSortKey(event.target.value)}>
            <option value="name">{pageText(locale, 'Name (unnamed last)', 'שם (ללא שם בסוף)')}</option>
            <option value="rules-desc">{pageText(locale, 'Rule count (high to low)', 'מספר כללים (גבוה לנמוך)')}</option>
            <option value="premium-desc">{pageText(locale, 'Premium (high to low)', 'מקדם (גבוה לנמוך)')}</option>
            <option value="premium-asc">{pageText(locale, 'Premium (low to high)', 'מקדם (נמוך לגבוה)')}</option>
            <option value="id">{pageText(locale, 'ID (A to Z)', 'מזהה (א-ת)')}</option>
          </SelectControl>
        </div>
        <Button
          className="secondary-button compact"
          type="button"
          variant="outlined"
          aria-expanded={showAdd}
          onClick={() => setShowAdd((value) => !value)}
        >
          {showAdd ? <X size={14} /> : <Plus size={14} />}
          {showAdd ? pageText(locale, 'Close', 'סגירה') : pageText(locale, 'Add advertiser', 'הוספת מפרסם')}
        </Button>
      </div>

      {showAdd && online && (
        <AddAdvertiserForm
          locale={locale}
          suggestedId={suggestedId}
          existingIds={existingIds}
          onCreate={writes.create}
          onCancel={() => setShowAdd(false)}
        />
      )}

      {loading && <div className="amz-empty">{pageText(locale, 'Loading advertisers...', 'טוען מפרסמים...')}</div>}

      {!loading && !online && (
        <div className="amz-empty">
          <Users size={22} />
          {pageText(locale, 'The Kairos API is unavailable. Advertisers cannot be shown.', 'ה־API של Kairos לא זמין. לא ניתן להציג מפרסמים.')}
        </div>
      )}

      {!loading && online && advertisers.length === 0 && (
        <div className="amz-empty">
          <Users size={22} />
          {pageText(locale, 'No advertisers yet. Add one to start building its management area.', 'אין עדיין מפרסמים. הוסף מפרסם כדי להתחיל לבנות את אזור הניהול שלו.')}
        </div>
      )}

      {!loading && online && advertisers.length > 0 && visible.length === 0 && (
        <div className="amz-empty">
          <span>{pageText(locale, 'No advertisers match your search or filter.', 'אין מפרסמים שתואמים את החיפוש או הסינון.')}</span>
          <Button className="secondary-button compact" type="button" variant="outlined" onClick={clearQuery}>
            {pageText(locale, 'Clear filters', 'ניקוי סינון')}
          </Button>
        </div>
      )}

      {!loading && online && visible.length > 0 && (
        <AdvertiserCardGrid rows={windowed} locale={locale} grouped={sortKey === 'name'} onOpen={setOpenId} />
      )}

      {!loading && online && visible.length > 0 && (hasActiveQuery || windowed.length < visible.length) && (
        <div className="amz-result-note clients-window-more" role="status">
          <span>
            {pageText(
              locale,
              `Showing ${windowed.length} of ${visible.length} matching records (${advertisers.length} total)`,
              `מוצגות ${windowed.length} מתוך ${visible.length} רשומות תואמות (${advertisers.length} בסך הכול)`,
            )}
          </span>
          {windowed.length < visible.length ? (
            <Button type="button" variant="outlined" className="clients-secondary" onClick={() => setVisibleCount((count) => count + ADVERTISER_WINDOW)}>
              {pageText(locale, 'Show the next records', 'הציגו את הרשומות הבאות')}
            </Button>
          ) : null}
        </div>
      )}

      <AdvertiserDetailDrawer
        row={openRow}
        open={Boolean(openRow)}
        locale={locale}
        scopeOptions={scopeOptions}
        onClose={() => setOpenId(null)}
        onSaveBaseline={writes.saveBaseline}
        onDelete={writes.remove}
        onCreateCondition={writes.createCondition}
        onUpdateCondition={writes.updateCondition}
        onDeleteCondition={writes.deleteCondition}
      />
    </section>
  );
}

export default AdvertiserRecordsPanel;
