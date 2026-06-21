import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from '@mui/material';
import { Info, Plus, RefreshCcw, Search, Users, X } from 'lucide-react';
import {
  normalizeRows,
  pageText,
  suggestNextId,
  toConditionPayload,
  toPayload,
} from './advertisers-helpers';
import {
  filterManaged,
  indexStats,
  managementSummary,
  mergeRowWithStats,
  sortManaged,
} from './advertiser-stats-helpers';
import AdvertiserStatCard from './AdvertiserStatCard';
import AdvertiserDetailDrawer from './AdvertiserDetailDrawer';
import AddAdvertiserForm from './AddAdvertiserForm';
import './advertiser-management.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || 'http://127.0.0.1:8000';

const FILTERS = [
  { key: 'all', en: 'All', he: 'הכול' },
  { key: 'with-rules', en: 'With scoped rules', he: 'עם כללים ממוקדים' },
  { key: 'custom-premium', en: 'Custom premium', he: 'מקדם מותאם' },
  { key: 'conflicts', en: 'Has conflicts', he: 'עם התנגשויות' },
];

function AdvertisersManager({ copy, locale, notify }) {
  const [advertisers, setAdvertisers] = useState([]);
  const [statsIndex, setStatsIndex] = useState(() => new Map());
  const [statusNote, setStatusNote] = useState('');
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState(true);
  const [statsError, setStatsError] = useState(false);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');
  const [sortKey, setSortKey] = useState('rules-desc');
  const [showAdd, setShowAdd] = useState(false);
  const [scopeOptions, setScopeOptions] = useState({});
  const [openId, setOpenId] = useState(null);

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
      await loadStats();
    } catch {
      setAdvertisers([]);
      setOnline(false);
    } finally {
      setLoading(false);
    }
  }, [loadStats]);

  useEffect(() => {
    loadAdvertisers();
  }, [loadAdvertisers]);

  const merged = useMemo(
    () => advertisers.map((row) => mergeRowWithStats(row, statsIndex)),
    [advertisers, statsIndex],
  );
  const summary = useMemo(() => managementSummary(merged), [merged]);
  const existingIds = useMemo(() => advertisers.map((row) => row.advertiser_id), [advertisers]);
  const suggestedId = useMemo(() => suggestNextId(advertisers), [advertisers]);
  const visible = useMemo(
    () => sortManaged(filterManaged(merged, { search, filter }), sortKey),
    [merged, search, filter, sortKey],
  );
  const openRow = useMemo(
    () => merged.find((row) => row.advertiser_id === openId) || null,
    [merged, openId],
  );
  const hasActiveQuery = Boolean(search.trim()) || filter !== 'all';

  async function putAdvertiser(draft) {
    const response = await fetch(`${API_BASE}/api/advertisers/${encodeURIComponent(draft.advertiser_id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(toPayload(draft)),
    });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return response;
  }

  async function handleSaveBaseline(draft) {
    try {
      await putAdvertiser(draft);
      notify(`Advertiser ${draft.advertiser_id} saved.`, `המפרסם ${draft.advertiser_id} נשמר.`);
      await loadAdvertisers();
    } catch (error) {
      notify(`Save failed for ${draft.advertiser_id} (${error.message}).`, `השמירה נכשלה עבור ${draft.advertiser_id} (${error.message}).`);
    }
  }

  async function handleCreate(draft) {
    try {
      const response = await fetch(`${API_BASE}/api/advertisers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ advertiser_id: draft.advertiser_id, ...toPayload(draft) }),
      });
      if (response.status === 409) {
        notify(`Advertiser ${draft.advertiser_id} already exists.`, `המפרסם ${draft.advertiser_id} כבר קיים.`);
        return false;
      }
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Advertiser ${draft.advertiser_id} created.`, `המפרסם ${draft.advertiser_id} נוצר.`);
      setShowAdd(false);
      await loadAdvertisers();
      return true;
    } catch (error) {
      notify(`Create failed (${error.message}).`, `היצירה נכשלה (${error.message}).`);
      return false;
    }
  }

  async function handleDelete(advertiserId) {
    try {
      const response = await fetch(`${API_BASE}/api/advertisers/${encodeURIComponent(advertiserId)}`, { method: 'DELETE' });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Advertiser ${advertiserId} deleted.`, `המפרסם ${advertiserId} נמחק.`);
      setOpenId(null);
      await loadAdvertisers();
    } catch (error) {
      notify(`Delete failed for ${advertiserId} (${error.message}).`, `המחיקה נכשלה עבור ${advertiserId} (${error.message}).`);
    }
  }

  async function handleCreateCondition(advertiserId, draft) {
    try {
      const ruleId = `rule_${Date.now().toString(36)}`;
      const response = await fetch(`${API_BASE}/api/advertisers/${encodeURIComponent(advertiserId)}/conditions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rule_id: ruleId, ...toConditionPayload(draft) }),
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Scoped rule added to ${advertiserId}.`, `כלל ממוקד נוסף ל${advertiserId}.`);
      await loadAdvertisers();
      return true;
    } catch (error) {
      notify(`Add rule failed for ${advertiserId} (${error.message}).`, `הוספת הכלל נכשלה עבור ${advertiserId} (${error.message}).`);
      return false;
    }
  }

  async function handleUpdateCondition(advertiserId, ruleId, draft) {
    try {
      const response = await fetch(`${API_BASE}/api/advertisers/${encodeURIComponent(advertiserId)}/conditions/${encodeURIComponent(ruleId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(toConditionPayload(draft)),
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Scoped rule saved for ${advertiserId}.`, `כלל ממוקד נשמר עבור ${advertiserId}.`);
      await loadAdvertisers();
      return true;
    } catch (error) {
      notify(`Save rule failed for ${advertiserId} (${error.message}).`, `שמירת הכלל נכשלה עבור ${advertiserId} (${error.message}).`);
      return false;
    }
  }

  async function handleDeleteCondition(advertiserId, ruleId) {
    try {
      const response = await fetch(`${API_BASE}/api/advertisers/${encodeURIComponent(advertiserId)}/conditions/${encodeURIComponent(ruleId)}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Scoped rule removed from ${advertiserId}.`, `כלל ממוקד הוסר מ${advertiserId}.`);
      await loadAdvertisers();
    } catch (error) {
      notify(`Delete rule failed for ${advertiserId} (${error.message}).`, `מחיקת הכלל נכשלה עבור ${advertiserId} (${error.message}).`);
    }
  }

  function clearQuery() {
    setSearch('');
    setFilter('all');
  }

  const cards = [
    { key: 'total', value: summary.total, en: 'Advertisers', he: 'מפרסמים' },
    { key: 'withRules', value: summary.withRules, en: 'With scoped rules', he: 'עם כללים ממוקדים' },
    { key: 'totalRules', value: summary.totalRules, en: 'Scoped rules total', he: 'סך כללים ממוקדים' },
    { key: 'conflicts', value: summary.conflicts, en: 'Conflicts flagged', he: 'התנגשויות שסומנו', warn: true },
  ];

  return (
    <section className="page-workspace">
      <div className="page-header">
        <div>
          <h1>{pageText(locale, 'Advertisers', 'מפרסמים')}</h1>
          <p>
            {pageText(
              locale,
              'A management area per advertiser: see at a glance whether it has rules and how many, the premium, and conflicts. Click any card to open its workspace.',
              'אזור ניהול לכל מפרסם: רואים במבט אחד האם יש לו כללים וכמה, את המקדם והתנגשויות. לחיצה על כרטיס פותחת את סביבת העבודה שלו.',
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
              <span className="amz-summary-value" dir="ltr">{card.value}</span>
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
              `Honest status: ${statusNote} Revenue and profitability are therefore shown as "-" until the daily path attribution lands.`,
              'מצב שקוף: המנוע השבועי אינו צורך את כללי המפרסמים; רק מסלול תמחור הספוטים היומי מתמחר מולם. לכן ההכנסה והרווחיות מוצגות כ-״-״ עד שייכנס ייחוס המסלול היומי.',
            )}
          </p>
        </div>
      )}

      <div className="amz-toolbar">
        <div className="amz-search">
          <Search size={15} />
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={pageText(locale, 'Search by ID or notes', 'חיפוש לפי מזהה או הערות')}
            aria-label={pageText(locale, 'Search advertisers', 'חיפוש מפרסמים')}
          />
        </div>
        <div className="amz-filter-chips" role="group" aria-label={pageText(locale, 'Filter advertisers', 'סינון מפרסמים')}>
          {FILTERS.map((entry) => (
            <button
              key={entry.key}
              type="button"
              className={`adv-chip${filter === entry.key ? ' active' : ''}`}
              aria-pressed={filter === entry.key}
              onClick={() => setFilter(entry.key)}
            >
              {pageText(locale, entry.en, entry.he)}
            </button>
          ))}
        </div>
        <div className="amz-sort">
          <label htmlFor="amz-sort-select">{pageText(locale, 'Sort', 'מיון')}</label>
          <select id="amz-sort-select" value={sortKey} onChange={(event) => setSortKey(event.target.value)}>
            <option value="rules-desc">{pageText(locale, 'Rule count (high to low)', 'מספר כללים (גבוה לנמוך)')}</option>
            <option value="premium-desc">{pageText(locale, 'Premium (high to low)', 'מקדם (גבוה לנמוך)')}</option>
            <option value="premium-asc">{pageText(locale, 'Premium (low to high)', 'מקדם (נמוך לגבוה)')}</option>
            <option value="id">{pageText(locale, 'ID (A to Z)', 'מזהה (א-ת)')}</option>
          </select>
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
          onCreate={handleCreate}
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
        <div className="amz-grid">
          {visible.map((row) => (
            <AdvertiserStatCard key={row.advertiser_id} row={row} locale={locale} onOpen={setOpenId} />
          ))}
        </div>
      )}

      {!loading && online && visible.length > 0 && hasActiveQuery && (
        <div className="amz-result-note">
          {pageText(locale, `Showing ${visible.length} of ${advertisers.length}`, `מוצגים ${visible.length} מתוך ${advertisers.length}`)}
        </div>
      )}

      <AdvertiserDetailDrawer
        row={openRow}
        open={Boolean(openRow)}
        locale={locale}
        scopeOptions={scopeOptions}
        onClose={() => setOpenId(null)}
        onSaveBaseline={handleSaveBaseline}
        onDelete={handleDelete}
        onCreateCondition={handleCreateCondition}
        onUpdateCondition={handleUpdateCondition}
        onDeleteCondition={handleDeleteCondition}
      />
    </section>
  );
}

export default AdvertisersManager;
