import React, { useEffect, useRef, useState } from 'react';
import { Tooltip } from '@mui/material';
import { Button } from '../../studio/actions';
import { Download } from 'lucide-react';
import { formatCurrency, formatMinutes, formatPercent, pageText } from '../../shell/format';
import { breakLengthLabel, breakPositionLabel, programTypeLabel } from '../../shell/labels';
import { normalizeRows } from '../../shell/plan-model';
import { buildCsv, downloadCsv } from '../../shell/downloads';
import { DataTable, PageHeader, StatusBadge } from '../../studio';
import ScheduleInspector from '../day/ScheduleInspector';
import BreakBoard from './BreakBoard';
import PodPage from './PodPage';
import './break-library.css';
import '../day/master-control-broadcast.css';

export function BreakLibraryPage({ breakLibrary, copy, locale, notify, onGlobalRefresh }) {
  const rows = normalizeRows(breakLibrary.breaks);
  // Click-to-open reuses the Schedule page's inspector drawer: every ranked row
  // carries the saved plan's segment_id, so the same /api/schedule/segment/{id}
  // detail and edit affordances open in place on this page. A row without a
  // segment id (older saved plan) gets an honest notice instead of a dead click.
  const [inspect, setInspect] = useState(null);
  // The state channel a covered-day link inside the break drawer below uses to
  // open a day in the pod section above, on the same page. A hash assignment
  // cannot do this, because the hash already names this page while the drawer
  // is open on it, so this is a real channel rather than a link with nowhere
  // to land. token forces the effect on PodPage to fire even for a day it is
  // already showing, so the click always answers with a scroll and a reload.
  const [podDayRequest, setPodDayRequest] = useState(null);
  const VIEWS = ['library', 'day', 'pod'];
  const [view, setView] = useState(() => {
    const addressed = typeof window === 'undefined' ? null : new URLSearchParams(window.location.search).get('breakView');
    return VIEWS.includes(addressed) ? addressed : 'library';
  });
  const viewRefs = useRef([]);

  function goView(next, { history = true, focus = false } = {}) {
    if (!VIEWS.includes(next)) return;
    setView(next);
    setInspect(null);
    if (typeof window !== 'undefined' && history) {
      const url = new URL(window.location.href);
      url.searchParams.set('breakView', next);
      window.history.pushState({ ...(window.history.state || {}), breakView: next }, '', `${url.pathname}${url.search}${url.hash}`);
    }
    if (focus) window.setTimeout(() => viewRefs.current[VIEWS.indexOf(next)]?.focus(), 0);
  }

  useEffect(() => {
    function restore() {
      const addressed = new URLSearchParams(window.location.search).get('breakView');
      setView(VIEWS.includes(addressed) ? addressed : 'library');
      setInspect(null);
    }
    window.addEventListener('popstate', restore);
    return () => window.removeEventListener('popstate', restore);
  }, []);

  const openPodDay = (day) => {
    setPodDayRequest({ day, token: Date.now() });
    goView('pod');
  };

  function moveView(event) {
    const current = VIEWS.indexOf(view);
    let next = current;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = VIEWS.length - 1;
    else if (event.key === 'ArrowRight') next = (current + (locale === 'he' ? -1 : 1) + VIEWS.length) % VIEWS.length;
    else if (event.key === 'ArrowLeft') next = (current + (locale === 'he' ? 1 : -1) + VIEWS.length) % VIEWS.length;
    else return;
    event.preventDefault();
    goView(VIEWS[next], { focus: true });
  }
  const openBreak = (row) => {
    if (row && row.segment_id) {
      setInspect({ segmentId: row.segment_id, channel: row.channel, day: row.day });
    } else if (notify) {
      notify('This row carries no segment id in the saved plan, so there is no detail to open.', 'לשורה זו אין מזהה מקטע בתוכנית השמורה, ולכן אין פירוט לפתיחה.');
    }
  };
  // Exports every ranked row, not only the visible grid page. Vocabulary values
  // are localized; numbers stay raw so spreadsheet sorting works.
  const exportLibraryCsv = () => {
    const csv = buildCsv([
      { label: pageText(locale, 'Status', 'סטטוס'), value: (row) => row.status },
      { label: pageText(locale, 'Channel', 'ערוץ'), value: (row) => row.channel },
      { label: pageText(locale, 'Date', 'תאריך'), value: (row) => row.date },
      { label: pageText(locale, 'Start time', 'שעת התחלה'), value: (row) => row.start_time },
      { label: pageText(locale, 'Programme type', 'סוג תוכנית'), value: (row) => programTypeLabel(row.program_type, locale) },
      { label: pageText(locale, 'Position', 'מיקום'), value: (row) => breakPositionLabel(row.position, locale) },
      { label: pageText(locale, 'Break type', 'סוג ברייק'), value: (row) => breakLengthLabel(row.break_type, locale) },
      { label: pageText(locale, 'Length (seconds)', 'אורך (שניות)'), value: (row) => row.total_break_time },
      { label: pageText(locale, 'Expected revenue (ILS)', 'הכנסה צפויה (ש"ח)'), value: (row) => Math.round(Number(row.predicted_revenue) || 0) },
      { label: pageText(locale, 'Expected retention (%)', 'שימור צפוי (%)'), value: (row) => (Number(row.predicted_retention || 0) * 100).toFixed(2) },
      { label: pageText(locale, 'Segment id', 'מזהה מקטע'), value: (row) => row.segment_id },
    ], rows);
    downloadCsv('kairos-break-library.csv', csv);
    if (notify) notify('Break library exported as CSV.', 'ספריית הברייקים יוצאה כ־CSV.');
  };
  return (
    <section className="page-workspace broadcast-library">
      <PageHeader
        locale={locale}
        titleEn="Break library"
        titleHe="ספריית ברייקים"
        bodyEn="Source: saved plan. Open a row for its schedule record; export includes every ranked break."
        bodyHe="מקור: התוכנית השמורה. פתיחת שורה מציגה את רשומת השיבוץ; הייצוא כולל את כל הברייקים המדורגים."
      />
      <nav className="break-local-nav" role="tablist" aria-label={pageText(locale, 'Break workspace views', 'תצוגות מרחב הברייק')} onKeyDown={moveView}>
        {VIEWS.map((id, index) => {
          const active = id === view;
          const labels = {
            pod: pageText(locale, 'Traffic pod', 'תוכן הברייק'),
            day: pageText(locale, 'Breaks by day', 'ברייקים לפי יום'),
            library: pageText(locale, 'Ranked library', 'ספרייה מדורגת'),
          };
          return (
            <Button
              key={id}
              ref={(node) => { viewRefs.current[index] = node; }}
              id={`break-view-tab-${id}`}
              type="button"
              role="tab"
              tabIndex={active ? 0 : -1}
              aria-selected={active}
              aria-controls={`break-view-panel-${id}`}
              onClick={() => goView(id)}
            >
              {labels[id]}
            </Button>
          );
        })}
      </nav>
      <div id={`break-view-panel-${view}`} role="tabpanel" aria-labelledby={`break-view-tab-${view}`} tabIndex={0}>
        {view === 'pod' && (
          <section className="page-panel">
            <PodPage locale={locale} notify={notify} requestedDay={podDayRequest} />
          </section>
        )}
        {view === 'day' && <BreakBoard locale={locale} notify={notify} onOpenPodDay={openPodDay} />}
        {view === 'library' && (
          <section className="page-panel">
            <div className="panel-head">
              <h2>{pageText(locale, 'Ranked break candidates', 'ברייקים מדורגים')}</h2>
              <div className="panel-head-tools">
                <span>{rows.length} {pageText(locale, 'breaks', 'ברייקים')}</span>
                <Tooltip title={pageText(locale, 'Exports every ranked break, not only the visible page.', 'הייצוא כולל את כל הברייקים המדורגים, לא רק את העמוד המוצג.')} arrow placement="bottom">
                  <span>
                    <Button className="secondary-button" type="button" variant="outlined" disabled={rows.length === 0} onClick={exportLibraryCsv}>
                      <Download size={16} aria-hidden="true" />
                      {pageText(locale, 'CSV export', 'ייצוא CSV')}
                    </Button>
                  </span>
                </Tooltip>
              </div>
            </div>
            <p className="row-open-hint">{pageText(locale, 'Selecting a row opens the break detail.', 'לחיצה על שורה פותחת את פרטי הברייק.')}</p>
            <DataTable
              locale={locale}
              fit
              pageSize={25}
              onRowClick={openBreak}
              emptyLabel={pageText(locale, 'No break candidates were found.', 'לא נמצאו ברייקים מועמדים.')}
              rows={rows}
              columns={[
                { key: 'status', label: pageText(locale, 'Status', 'סטטוס'), status: true, minWidth: 104, flex: 0.55, render: (row) => <StatusBadge status={row.status} locale={locale} mode="cell" /> },
                { key: 'channel', label: pageText(locale, 'Channel', 'ערוץ') },
                { key: 'date', label: pageText(locale, 'Airing', 'שידור'), numeric: true, render: (row) => [row.date, row.start_time].filter(Boolean).join(' ') || '-' },
                { key: 'program_type', label: pageText(locale, 'Programme type', 'סוג תוכנית'), render: (row) => programTypeLabel(row.program_type, locale) },
                { key: 'position', label: pageText(locale, 'Position', 'מיקום'), render: (row) => breakPositionLabel(row.position, locale) },
                { key: 'break_type', label: pageText(locale, 'Type', 'סוג'), render: (row) => breakLengthLabel(row.break_type, locale) },
                { key: 'total_break_time', label: pageText(locale, 'Length', 'אורך'), render: (row) => formatMinutes(row.total_break_time, locale) },
                { key: 'predicted_revenue', label: pageText(locale, 'Revenue', 'הכנסה'), render: (row) => formatCurrency(row.predicted_revenue, locale) },
                { key: 'predicted_retention', label: pageText(locale, 'Retention', 'שימור'), render: (row) => formatPercent(Number(row.predicted_retention || 0) * 100, locale) },
              ]}
            />
          </section>
        )}
      </div>
      {view === 'library' && inspect && (
        <ScheduleInspector
          segmentId={inspect.segmentId}
          channel={inspect.channel}
          day={inspect.day}
          locale={locale}
          notify={notify}
          onClose={() => setInspect(null)}
          onGlobalRefresh={onGlobalRefresh}
        />
      )}
    </section>
  );
}

export default BreakLibraryPage;
