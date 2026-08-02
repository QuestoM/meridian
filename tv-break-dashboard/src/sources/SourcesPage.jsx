import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Button } from '@mui/material';
import { RefreshCcw } from 'lucide-react';
import { PageHeader } from '../shell/primitives';
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

export function SourcesPage({ view: initialView, files, overview, reports, locale, notify, onGlobalRefresh }) {
  const [view, setViewState] = useState(VIEWS.includes(initialView) ? initialView : 'inputs');
  const [filter, setFilterState] = useState(filterFromLocation);
  const [status, setStatus] = useState({ loading: true, online: true, body: { inputs: [] } });
  const [ownReports, setOwnReports] = useState(null);
  const [highlight, setHighlight] = useState('');
  // The account, for the fallback only: the endpoint is the authority and the
  // status payload carries its own can_edit with the reason the server would
  // refuse with.
  const [session, setSession] = useState(null);

  useEffect(() => {
    let active = true;
    fetchSession().then((result) => {
      if (active && result.ok) setSession(result.session);
    });
    return () => {
      active = false;
    };
  }, []);

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
    setViewState(next);
    if (typeof window === 'undefined') return;
    const hash = VIEW_HASH[next];
    if (!hash || decodeURIComponent(window.location.hash.replace(/^#/, '')) === hash) return;
    window.location.hash = encodeURIComponent(hash);
  }, []);

  const setFilter = useCallback((next) => {
    setFilterState(next);
    if (typeof window === 'undefined' || !window.history) return;
    const params = new URLSearchParams(window.location.search);
    if (next === 'all') params.delete('source');
    else params.set('source', next);
    const query = params.toString();
    // replaceState, not a navigation: the hash the shell routes on must not
    // move, and stepping back through seven filter clicks is nobody's history.
    window.history.replaceState(null, '', `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`);
  }, []);

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
        bodyEn="Every input a run reads, what the engine is actually reading right now, and the reports built from it."
        bodyHe="כל קלט שהרצה קוראת, מה שהמנוע קורא בפועל כרגע, והדוחות שנבנים ממנו."
        action={
          <Button className="secondary-button compact" type="button" variant="outlined" onClick={reload}>
            <RefreshCcw size={14} />
            {locale === 'he' ? 'רענון' : 'Refresh'}
          </Button>
        }
      />

      <div className="surface-toolbar no-print">
        <div className="toolbar-left" role="tablist">
          {VIEWS.map((key) => (
            <Button
              key={key}
              className={view === key ? 'segmented active' : 'segmented'}
              type="button"
              variant="outlined"
              role="tab"
              aria-selected={view === key}
              onClick={() => setView(key)}
            >
              {label(VIEW_LABELS, key, locale)}
            </Button>
          ))}
        </div>
      </div>

      {view === 'inputs' && status.loading ? <p className="sources-note">{text('loading', locale)}</p> : null}
      {view === 'inputs' && !status.loading && !status.online ? (
        <p className="sources-note">{text('offline', locale)}</p>
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
    </section>
  );
}

export default SourcesPage;
