import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '../studio/actions';
import { RefreshCcw } from 'lucide-react';
import { PageHeader } from '../studio';
import { WALLS, fetchSession, payloadCanEdit } from '../session';
import { FILTER_ORDER as FILTERS, VIEWS, VIEW_HASH, VIEW_LABELS, label, text } from './sources-copy';
import { fetchReports, fetchUploadStatus } from './sources-api';
import InputsView from './InputsView';
import SourceFilesView from './SourceFilesView';
import DownloadsView from './DownloadsView';
import './sources.css';
import './sources-card.css';
import './sources-tables.css';
import './sources-findings.css';
import './sources-stored.css';
import './studio-ledger-sources.css';
import './studio-ledger-sources-ledger.css';
import './studio-ledger-sources-overlays.css';

// Sources is one destination with three views, and the view is a control in
// the content rather than a second destination. Two navigation entries lead
// here, so each one selects the view it names and the address stays in step
// with what is on screen.
// A state is also a place. The rail entry lives in the hash, which the shell
// owns, so the filter travels in the query string beside the axis parameter
// the shell already uses there. Reading it back on mount is what makes a
// filtered view returnable and sendable rather than a click somebody else has
// to repeat.
function filterFromLocation() {
  if (typeof window === 'undefined') return 'all';
  const requested = new URLSearchParams(window.location.search).get('source');
  return FILTERS.includes(requested) ? requested : 'all';
}

function viewFromLocation(fallback = 'inputs') {
  if (typeof window === 'undefined') return fallback;
  const requested = new URLSearchParams(window.location.search).get('sourceView');
  return VIEWS.includes(requested) ? requested : fallback;
}

// A file named somewhere else in the product, arriving as a request to show it.
//
// It travels by NAME because the surfaces that print it — a delivery basis, a
// money basis, a pacing drill — hold a filename and nothing else. Resolving that
// name to the input it belongs to happens HERE, against the upload status this
// screen already fetches, so the mapping lives in one place rather than being
// re-derived by every caller from a path convention.
//
// Matched on the basename in both directions: the ledger records
// "Wally_Prime_Reshet_Example_2025-04-27.csv" while the status carries
// "data/daily_input/Wally_Prime_Reshet_Example_2025-04-27.csv", and either may
// be the one asking.
function fileFromLocation() {
  if (typeof window === 'undefined') return '';
  return (new URLSearchParams(window.location.search).get('sourceFile') || '').trim();
}

function baseName(value) {
  return String(value || '').split('/').pop();
}

function kindForFile(inputs, wanted) {
  if (!wanted) return '';
  const target = baseName(wanted);
  const hit = (inputs || []).find(
    (input) => baseName(input.path) === target || baseName(input.file) === target);
  return hit ? String(hit.kind || '') : '';
}

export function SourcesPage({ view: initialView, files, overview, reports, locale, notify, onGlobalRefresh }) {
  const fallbackView = VIEWS.includes(initialView) ? initialView : 'inputs';
  const [view, setViewState] = useState(() => viewFromLocation(fallbackView));
  const [filter, setFilterState] = useState(filterFromLocation);
  const [wantedFile, setWantedFile] = useState(fileFromLocation);
  const [status, setStatus] = useState({ loading: true, online: true, body: { inputs: [] } });
  const [ownReports, setOwnReports] = useState(null);
  const [highlight, setHighlight] = useState('');
  // The account, for the fallback only: the endpoint is the authority and the
  // status payload carries its own can_edit with the reason the server would
  // refuse with.
  const [session, setSession] = useState(null);
  const tabsRef = useRef([]);

  useEffect(() => {
    let active = true;
    fetchSession().then((result) => {
      if (active && result.ok) setSession(result.session);
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    function syncFromAddress() {
      setViewState(viewFromLocation(fallbackView));
      setFilterState(filterFromLocation());
      setWantedFile(fileFromLocation());
    }
    window.addEventListener('popstate', syncFromAddress);
    return () => window.removeEventListener('popstate', syncFromAddress);
  }, [fallbackView]);

  const loadStatus = useCallback(async () => {
    setStatus((current) => ({ ...current, loading: true }));
    const result = await fetchUploadStatus();
    setStatus({ loading: false, online: result.online, body: result.status || { inputs: [] } });
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  // The reports payload arrives as a prop from one door and not from the other,
  // so the destination fetches it itself when it is asked for the view that
  // needs it and was not given one.
  useEffect(() => {
    if (view !== 'downloads' || reports || ownReports) return;
    fetchReports().then((result) => setOwnReports(result.reports || { reports: [] }));
  }, [view, reports, ownReports]);

  const setView = useCallback((next) => {
    if (!VIEWS.includes(next)) return;
    setViewState(next);
    if (typeof window === 'undefined') return;
    const hash = VIEW_HASH[next];
    const params = new URLSearchParams(window.location.search);
    params.set('sourceView', next);
    const query = params.toString();
    const nextUrl = `${window.location.pathname}${query ? `?${query}` : ''}${hash ? `#${encodeURIComponent(hash)}` : window.location.hash}`;
    const hashChanged = hash && decodeURIComponent(window.location.hash.replace(/^#/, '')) !== hash;
    window.history.pushState({ workspace: 'sources', section: next }, '', nextUrl);
    if (hashChanged) window.dispatchEvent(new HashChangeEvent('hashchange'));
  }, []);

  const setFilter = useCallback((next) => {
    setFilterState(next);
    if (typeof window === 'undefined' || !window.history) return;
    const params = new URLSearchParams(window.location.search);
    if (next === 'all') params.delete('source');
    else params.set('source', next);
    const query = params.toString();
    window.history.pushState({ workspace: 'sources', filter: next }, '', `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`);
  }, []);

  const onTabKeyDown = useCallback((event, index) => {
    let next = index;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = VIEWS.length - 1;
    else if (event.key === 'ArrowRight') next = (index + (locale === 'he' ? -1 : 1) + VIEWS.length) % VIEWS.length;
    else if (event.key === 'ArrowLeft') next = (index + (locale === 'he' ? 1 : -1) + VIEWS.length) % VIEWS.length;
    else return;
    event.preventDefault();
    setView(VIEWS[next]);
    tabsRef.current[next]?.focus();
  }, [locale, setView]);

  const openFile = useCallback((path) => {
    setHighlight(String(path || ''));
    setView('files');
  }, [setView]);

  const reload = useCallback(async () => {
    await loadStatus();
    if (onGlobalRefresh) onGlobalRefresh();
  }, [loadStatus, onGlobalRefresh]);

  const gate = useMemo(
    () => payloadCanEdit(status.body, session, WALLS.readOnlyRole),
    [status.body, session],
  );

  return (
    <section className="page-workspace sources-page">
      <PageHeader
        locale={locale}
        titleEn="Sources"
        titleHe="מקורות"
        bodyEn="Inputs the engine reads now and the reports built from them."
        bodyHe="קלטים שהמנוע קורא כרגע והדוחות שנבנים מהם."
        action={
          <Button className="secondary-button compact" type="button" variant="outlined" onClick={reload}>
            <RefreshCcw size={14} />
            {locale === 'he' ? 'רענון' : 'Refresh'}
          </Button>
        }
      />

      <div className="surface-toolbar no-print">
        <div className="toolbar-left" role="tablist" aria-label={locale === 'he' ? 'מדורי מקורות' : 'Source sections'}>
          {VIEWS.map((key, index) => (
            <Button
              ref={(node) => { tabsRef.current[index] = node; }}
              key={key}
              id={`sources-tab-${key}`}
              className={view === key ? 'segmented active' : 'segmented'}
              type="button"
              variant="outlined"
              role="tab"
              aria-selected={view === key}
              aria-controls={`sources-panel-${key}`}
              tabIndex={view === key ? 0 : -1}
              onClick={() => setView(key)}
              onKeyDown={(event) => onTabKeyDown(event, index)}
            >
              {label(VIEW_LABELS, key, locale)}
            </Button>
          ))}
        </div>
      </div>

      <div id={`sources-panel-${view}`} role="tabpanel" aria-labelledby={`sources-tab-${view}`} tabIndex={0}>
        {view === 'inputs' && status.loading ? <p className="sources-note" role="status">{text('loading', locale)}</p> : null}
        {view === 'inputs' && !status.loading && !status.online ? (
          <p className="sources-note" role="alert">{text('offline', locale)}</p>
        ) : null}
        {view === 'inputs' && !status.loading && status.online ? (
          <InputsView
            status={status.body}
            locale={locale}
            canEdit={gate.canEdit}
            canEditReason={gate.reason}
            filter={filter}
            onFilter={setFilter}
            onOpenFile={openFile}
            onReload={reload}
            notify={notify}
            focusKind={kindForFile(status.body.inputs, wantedFile)}
            focusFile={wantedFile}
          />
        ) : null}

        {view === 'files' ? (
          <SourceFilesView
            files={files}
            inputs={status.body.inputs}
            locale={locale}
            highlight={highlight}
          />
        ) : null}

        {view === 'downloads' ? (
          <DownloadsView
            reports={reports || ownReports || { reports: [] }}
            files={files}
            overview={overview}
            locale={locale}
            notify={notify}
          />
        ) : null}
      </div>
    </section>
  );
}

export default SourcesPage;
