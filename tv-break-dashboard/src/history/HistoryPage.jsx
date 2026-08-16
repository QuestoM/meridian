import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '../studio/actions';
import { Camera, RefreshCcw, Search } from 'lucide-react';
import { formatNumber, pageText } from '../shell/format';
import { Figure } from '../shell/bidi';
import { InputControl, Pressable, SelectControl } from '../studio/dom-controls';
import { ANONYMOUS_SESSION, WALLS, fetchSession, payloadCanEdit } from '../session.js';
import HistoryDetail from './HistoryDetail';
import { ReachDays, ReachEmpty, ReachEmptyPage, ReachMissed, ReachPager, ReachStart } from './HistoryReach';
import HistoryRunsSource from './HistoryRunsSource';
import HistorySince from './HistorySince';
import HistoryTimeline from './HistoryTimeline';
import { fetchTimeline, saveRestorePoint } from './history-api';
import { changesSourceLine, emptyWindow } from './history-reach';
import { foldPreviews, foldSize, matchesSearch } from './history-fold';
import { refusedTabLine } from './history-refused';
import { runsCountLine, runsCounted, runsSourceState } from './history-runs';
import { pageCoveredLine } from './history-scope';
import { useHistoryKindNavigation } from './use-history-page-navigation';
import {
  DEFAULT_LIMIT,
  WIDE_LIMIT,
  addressQuery,
  dayOfAddress,
  missedReason,
  pointAddress,
} from './history-address';
import {
  KEY_HINTS,
  KIND_HINTS,
  KIND_LABELS,
  actorLabel,
  historyPlace,
  pair,
  readAddress,
  writeAddress,
} from './history-labels';
import './history.css';
import './studio-ledger-history.css';

// One timeline over the four records the product already keeps and never showed
// together, with content-level filters and an addressable opened entry.
// Nothing here computes a figure. Every number on this surface was recorded by
// the engine, version store or request recorder; unreadable records say so.

export default function HistoryPage({ locale, notify }) {
  const [session, setSession] = useState(ANONYMOUS_SESSION);
  const [state, setState] = useState('loading');
  const [body, setBody] = useState(null);
  const [error, setError] = useState('');
  // A restore-point address opens on the points list; the unfiltered live page
  // can move beyond its 500-entry reach before a shared link is opened.
  const [actor, setActor] = useState('');
  const [needle, setNeedle] = useState('');
  // Inclusive broadcast days move by date; the cursor advances one page when a
  // day itself holds more than a page.
  const [fromDay, setFromDay] = useState('');
  const [untilDay, setUntilDay] = useState('');
  const [before, setBefore] = useState('');
  // Addressed entries get the wide first read so reachable links resolve.
  const [limit, setLimit] = useState(() => (readAddress() ? WIDE_LIMIT : DEFAULT_LIMIT));
  const [selectedId, setSelectedId] = useState(readAddress);
  const [saving, setSaving] = useState(false);
  const [pointLabel, setPointLabel] = useState('');
  const [pointOpen, setPointOpen] = useState(false);
  // Remember the filters behind the loaded body so a pending link is not judged
  // against the previous response.
  const [loaded, setLoaded] = useState(null);
  const searchRef = useRef(null);
  const listRef = useRef(null);
  const reading = useRef(0);
  // Capture the arrival address before row selection rewrites the URL.
  const requested = useRef(readAddress());
  const detailDismissed = useRef(false);
  const { chooseKind, kind, kindTabsRef, onKindTabKeyDown, setKind } =
    useHistoryKindNavigation({ locale, requested, setBefore, setSelectedId });

  useEffect(() => {
    let active = true;
    fetchSession().then((result) => {
      if (active && result.session) setSession(result.session);
    });
    return () => { active = false; };
  }, []);

  // Segmented dates can fire overlapping reads; only the newest may commit.
  const load = useCallback(async () => {
    const ticket = (reading.current += 1);
    setState('loading');
    const result = await fetchTimeline({ limit, kind, actor, since: fromDay, until: untilDay, before });
    if (ticket !== reading.current) return;
    if (!result.ok) {
      setState('error');
      setError(result.error);
      return;
    }
    setBody(result.data);
    setLoaded({ limit, kind, actor, fromDay, untilDay, before });
    setError('');
    setState('ready');
  }, [limit, kind, actor, fromDay, untilDay, before]);

  useEffect(() => { load(); }, [load]);

  // Changing the result set resets its cursor to the newest end.
  const setDays = useCallback((from, until) => {
    setBefore('');
    setFromDay(from);
    setUntilDay(until);
  }, []);

  const entries = useMemo(() => {
    const all = body && Array.isArray(body.entries) ? body.entries : [];
    const text = needle.trim().toLowerCase();
    return foldPreviews(all.filter((entry) => matchesSearch(entry, text, locale)));
  }, [body, needle, locale]);

  const selectedIndex = entries.findIndex((entry) => entry.id === selectedId);
  const selected = selectedIndex >= 0 ? entries[selectedIndex] : null;

  // Drop every content filter without touching the separately controlled days.
  const clearFilters = useCallback(() => { setBefore(''); setKind(''); setActor(''); setNeedle(''); }, []);

  // A row the reader picks answers whatever was asked before it, so the pending
  // address is dropped with the same act and the default below reopens.
  const choose = useCallback((id) => {
    requested.current = '';
    detailDismissed.current = !id;
    setSelectedId(id);
    writeAddress(id);
  }, []);

  // The newest entry opens by itself, so the destination answers "what changed
  // and by whom" without overriding an address or an explicit close.
  useEffect(() => {
    if (selected || !entries.length || requested.current) return;
    if (detailDismissed.current) return;
    setSelectedId(entries[0].id);
    writeAddress(entries[0].id);
  }, [entries, selected]);

  useEffect(() => {
    if (selected && requested.current === selected.id) requested.current = '';
  }, [selected]);

  const step = useCallback((direction) => {
    if (!entries.length) return;
    const next = Math.min(entries.length - 1, Math.max(0, (selectedIndex < 0 ? -1 : selectedIndex) + direction));
    choose(entries[next].id);
    const row = listRef.current && listRef.current.querySelector(`[data-index="${next}"]`);
    if (row && row.scrollIntoView) row.scrollIntoView({ block: 'nearest' });
  }, [entries, selectedIndex, choose]);

  // The one link that answers "how to put it back": inside a restore, the point
  // it came from and the point that undoes it. The reader is most often standing
  // in the Restore filter when they click it, and what they are asking for is a
  // restore point, which that filter excludes, so the link moves the list to
  // where the answer lives before it selects. It narrows rather than clears, for
  // the reason history-address.js measures beside addressQuery: on real volume an
  // unfiltered page cannot promise to hold a point and the list of points can.
  // The tab lights up, so the list did not change by magic.
  const openVersion = useCallback((versionId) => {
    const address = pointAddress(versionId);
    if (!address) return;
    detailDismissed.current = false;
    const query = addressQuery(address);
    requested.current = address;
    setKind(query.kind);
    setActor(query.actor);
    setNeedle(query.needle);
    setLimit(query.limit);
    setDays('', '');
    setSelectedId(address);
    writeAddress(address);
  }, [setDays]);

  useEffect(() => {
    function onKey(event) {
      const tag = String(event.target && event.target.tagName ? event.target.tagName : '').toLowerCase();
      const typing = tag === 'input' || tag === 'textarea' || tag === 'select';
      if (event.key === '/' && !typing) {
        event.preventDefault();
        if (searchRef.current) searchRef.current.focus();
        return;
      }
      if (typing) {
        if (event.key === 'Escape' && event.target.blur) event.target.blur();
        return;
      }
      if (event.key === 'j' || event.key === 'ArrowDown') { event.preventDefault(); step(1); }
      else if (event.key === 'k' || event.key === 'ArrowUp') { event.preventDefault(); step(-1); }
      else if (event.key === 'Escape') choose('');
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [step, choose]);

  const gate = payloadCanEdit(body || {}, session, WALLS.readOnlyRole);

  const savePoint = useCallback(async () => {
    setSaving(true);
    const result = await saveRestorePoint(pointLabel.trim());
    setSaving(false);
    if (!result.ok) {
      notify(`The restore point was not saved. ${result.error}`, `נקודת השחזור לא נשמרה. ${result.error}`);
      return;
    }
    setPointLabel('');
    setPointOpen(false);
    notify('A restore point was saved.', 'נקודת שחזור נשמרה.');
    load();
  }, [pointLabel, notify, load]);

  const counts = (body && body.counts) || {};
  const total = (body && body.total) || 0;
  // The tabs count inside the day window, so a tab never prints a figure over a set the reader is not looking at.
  const windowTotal = body && body.window_total !== undefined ? body.window_total : total;
  // How many of the changes in that window the server refused. The Change tab counts attempts, because an
  // attempt is what a person comes here to find, and this stops that figure reading as a count of changes.
  const refused = Number((((body && body.outcomes) || {}).refused) || 0);
  const windowed = Boolean(fromDay || untilDay);
  const shown = entries.length;
  const covered = entries.reduce((sum, entry) => sum + foldSize(entry), 0);
  // Addressed entries outside this response keep their request visible.
  const wanted = requested.current;
  const settled = Boolean(loaded) && loaded.limit === limit && loaded.kind === kind
    && loaded.actor === actor && loaded.untilDay === untilDay && loaded.before === before;
  const addressMissed = state === 'ready' && settled && wanted && !entries.some((entry) => entry.id === wanted);
  // Which of the four true things to say when the asked-for entry is not here, and whether a control can fix it.
  const pagedOut = ((body && body.matched) || 0) > limit;
  const missed = missedReason({ wanted, kind, actor, needle, pagedOut });
  // A change, a restore and an account event are addressed by their own stamp,
  // so an entry the page cannot reach can be answered with the day it is on. The
  // jump is withheld once the list already stands on that day, because a control
  // that would change nothing is worse than no control.
  const onThatDay = missed === 'paged_out' ? dayOfAddress(wanted) : '';
  const missedDay = onThatDay === fromDay && onThatDay === untilDay ? '' : onThatDay;
  const sources = (body && body.sources) || {};
  const runScope = (body && body.run_scope) || {};
  const runsState = runsSourceState(sources);
  // A withheld or unreadable run log is unknown, never zero.
  const counted = !body || runsCounted(runsState);
  const runsHint = counted ? '' : pageText(locale, ...runsCountLine(runsState));
  const runsBlocked = kind === 'run' && !counted;
  // Empty-window reasons come from the response's own counts.
  const emptied = emptyWindow(body, { kind, actor, needle: needle.trim() });

  return (
    <section className="page-workspace hist-workspace">
      <div className="page-header hist-header">
        <div className="hist-heading-copy">
          <h1>{historyPlace(locale)}</h1>
          <p>{pageText(locale, 'Recorded changes, model runs, restore points and access events.', 'שינויים, הרצות מודל, נקודות שחזור ואירועי גישה שנרשמו במערכת.')}</p>
        </div>
        <div className="hist-header-meta">
          <dl className="card card-dense card-body hist-head-figures">
            <div><dt>{pageText(locale, 'Recorded entries', 'רשומות מתועדות')}</dt><dd><Figure>{body ? formatNumber(total, locale) : pageText(locale, 'Pending', 'בהמתנה')}</Figure></dd></div>
            <div><dt>{pageText(locale, 'Current window', 'בחלון הנוכחי')}</dt><dd><Figure>{body ? formatNumber(windowTotal, locale) : pageText(locale, 'Pending', 'בהמתנה')}</Figure></dd></div>
            <div><dt>{pageText(locale, 'Operators', 'מפעילים')}</dt><dd><Figure>{body ? formatNumber((body.actors || []).length, locale) : pageText(locale, 'Pending', 'בהמתנה')}</Figure></dd></div>
          </dl>
          <div className="hist-keys" aria-label={pageText(locale, 'Keyboard', 'מקלדת')}>
            {KEY_HINTS.map(([key, en, he]) => (
              <span className="hist-key" key={key}>
                <kbd><Figure>{key}</Figure></kbd>
                {pageText(locale, en, he)}
              </span>
            ))}
          </div>
        </div>
      </div>

      <HistorySince locale={locale} landing={body && body.attestation} onShow={(since) => { clearFilters(); setDays(since, ''); }} />

      <div className="card card-dense card-body hist-toolbar">
        <div className="hist-kinds" role="tablist" aria-label={pageText(locale, 'Filter', 'סינון')}>
          <Pressable
            ref={(node) => { kindTabsRef.current[0] = node; }}
            type="button"
            role="tab"
            id="history-tab-all"
            aria-selected={kind === ''}
            aria-controls="history-results-panel"
            tabIndex={kind === '' ? 0 : -1}
            className={`hist-tab${kind === '' ? ' on' : ''}`}
            onClick={() => chooseKind('')}
            onKeyDown={(event) => onKindTabKeyDown(event, 0)}
          >
            {pageText(locale, 'Everything', 'הכול')}
            <span className="hist-tab-count"><Figure>{body ? windowTotal : pageText(locale, 'Pending', 'בהמתנה')}</Figure></span>
          </Pressable>
          {Object.keys(KIND_LABELS).map((name, index) => (
            <Pressable
              ref={(node) => { kindTabsRef.current[index + 1] = node; }}
              type="button"
              role="tab"
              key={name}
              id={`history-tab-${name}`}
              aria-selected={kind === name}
              aria-controls="history-results-panel"
              tabIndex={kind === name ? 0 : -1}
              title={name === 'run' && runsHint ? runsHint : pair(KIND_HINTS, name, locale)}
              className={`hist-tab${kind === name ? ' on' : ''}`}
              onClick={() => chooseKind(name)}
              onKeyDown={(event) => onKindTabKeyDown(event, index + 1)}
            >
              <span className={`hist-dot k-${name}`} aria-hidden="true" />
              {pair(KIND_LABELS, name, locale)}
              {name === 'run' && runsHint ? (
                <span className="hist-tab-count unknown" aria-label={runsHint}>?</span>
              ) : (
                <span className="hist-tab-count"><Figure>{body ? (counts[name] || 0) : pageText(locale, 'Pending', 'בהמתנה')}</Figure></span>
              )}
              {name === 'change' && refused ? (
                <span className="hist-tab-refused" title={pageText(locale, ...refusedTabLine(refused))}><Figure>{refused}</Figure></span>
              ) : null}
            </Pressable>
          ))}
        </div>
        <div className="hist-controls">
          <label className="hist-search">
            <Search size={14} aria-hidden="true" />
            <InputControl
              ref={searchRef}
              value={needle}
              onChange={(event) => setNeedle(event.target.value)}
              placeholder={pageText(locale, 'Search this list', 'חיפוש ברשימה')}
              aria-label={pageText(locale, 'Search this list', 'חיפוש ברשימה')}
            />
          </label>
          <label className="hist-select">
            <span>{pageText(locale, 'Operator', 'מפעיל')}</span>
            <SelectControl value={actor} onChange={(event) => { setBefore(''); setActor(event.target.value); }}>
              <option value="">{pageText(locale, 'Everyone', 'כולם')}</option>
              {((body && body.actors) || []).map((name) => <option key={name} value={name}>{actorLabel(name, locale)}</option>)}
            </SelectControl>
          </label>
          <ReachDays locale={locale} from={fromDay} until={untilDay} onDays={setDays} />
          <Button type="button" variant="outlined" size="small" onClick={load} disabled={state === 'loading'} startIcon={<RefreshCcw size={14} />}>
            {pageText(locale, 'Refresh', 'רענון')}
          </Button>
        </div>
        {gate.canEdit ? (
          <div className="hist-point">
            {pointOpen ? (
              <div className="hist-point-form">
                <InputControl
                  value={pointLabel}
                  onChange={(event) => setPointLabel(event.target.value)}
                  maxLength={120}
                  placeholder={pageText(locale, 'Name this restore point', 'שם לנקודת השחזור')}
                  aria-label={pageText(locale, 'Restore point name', 'שם נקודת השחזור')}
                  disabled={saving}
                />
                <Button variant="contained" size="small" onClick={savePoint} disabled={saving}>
                  {saving ? pageText(locale, 'Saving', 'שומר') : pageText(locale, 'Save the point', 'שמירת הנקודה')}
                </Button>
                <Button variant="text" size="small" onClick={() => { setPointOpen(false); setPointLabel(''); }} disabled={saving}>
                  {pageText(locale, 'Cancel', 'ביטול')}
                </Button>
              </div>
            ) : (
              <Pressable type="button" className="hist-link" onClick={() => setPointOpen(true)}>
                <Camera size={13} aria-hidden="true" />
                {pageText(locale, 'Save a restore point now', 'שמירת נקודת שחזור עכשיו')}
              </Pressable>
            )}
          </div>
        ) : (
          <span className="hist-block" role="note">{gate.reason}</span>
        )}
      </div>

      <div id="history-results-panel" role="tabpanel" aria-labelledby={`history-tab-${kind || 'all'}`} tabIndex={0}>
        {state === 'loading' ? <p className="hist-empty" role="status">{pageText(locale, 'Reading the record', 'קורא את הרישום')}</p> : null}
        {state === 'error' ? (
          <p className="hist-empty warn" role="alert">{pageText(locale, `History could not be read. ${error}`, `לא ניתן לקרוא את ההיסטוריה. ${error}`)}</p>
        ) : null}

      {addressMissed ? (
        <ReachMissed locale={locale} missed={missed} points={counts.restore_point || 0} limit={limit} wide={WIDE_LIMIT} day={missedDay}
          onClear={clearFilters} onWide={() => setLimit(WIDE_LIMIT)} onDay={() => setDays(missedDay, missedDay)} />
      ) : null}

      {/* An empty run list has one true cause and it is not the filters: the source
          is withheld or unreadable, so it says what the footer says and opens the same door. */}
      {state === 'ready' && !shown ? (
        <p className={`hist-empty${runsBlocked ? ' warn' : ''}`}>
          {runsBlocked ? (
            <HistoryRunsSource locale={locale} state={runsState} records={(sources.runs || {}).records} channel={runScope.scope_channel} />
          ) : null}
          {!runsBlocked && windowed ? (
            <ReachEmpty locale={locale} empty={emptied} onClear={clearFilters} onNewest={() => setDays('', '')} />
          ) : null}
          {!runsBlocked && !windowed ? (
            <ReachEmptyPage locale={locale} body={body} kind={kind} actor={actor} needle={needle.trim()} limit={limit} wide={WIDE_LIMIT}
              onClear={clearFilters} onActor={(name) => { setBefore(''); setNeedle(''); setActor(name); }}
              onWide={() => setLimit(WIDE_LIMIT)} onOlder={() => setBefore((body && body.next_before) || '')} onNewest={() => setBefore('')} />
          ) : null}
        </p>
      ) : null}

      {state === 'ready' && shown ? (
        <div className="hist-body">
          <HistoryTimeline
            entries={entries}
            locale={locale}
            selectedId={selectedId}
            onSelect={(entry) => choose(entry.id)}
            listRef={listRef}
          />
          <HistoryDetail
            entry={selected}
            locale={locale}
            position={selectedIndex + 1}
            total={shown}
            canEdit={gate.canEdit}
            canEditReason={gate.reason}
            notify={notify}
            onChanged={load}
            onStep={step}
            onClose={() => choose('')}
            onOpenVersion={openVersion}
          />
        </div>
      ) : null}

      {state === 'ready' ? (
        <footer className="hist-provenance">
          <span>{pageText(locale, `Showing ${shown} rows over ${covered} of ${formatNumber(total, 'en')} recorded entries.`, `מוצגות ${shown} שורות על ${covered} מתוך ${formatNumber(total, 'he')} רשומות.`)}</span>
          {/* How far back that record goes. "0 of 5,396" is a true sentence about a
              record that stops five hours ago, and alone it reads as an answer. */}
          <ReachStart locale={locale} body={body} />
          <ReachPager locale={locale} body={body} searching={Boolean(needle.trim())} onOlder={setBefore} onNewest={() => setBefore('')} />
          <span>{pageText(locale, `Restore points: ${(sources.restore_points || {}).records || 0}.`, `נקודות שחזור: ${(sources.restore_points || {}).records || 0}.`)}</span>
          <span>{pageText(locale, ...changesSourceLine(sources.changes))}</span>
          <HistoryRunsSource locale={locale} state={runsState} records={(sources.runs || {}).records} channel={runScope.scope_channel} />
          {body && body.scope === 'self' ? <span>{pageText(locale, ...pageCoveredLine(body))}</span> : null}
          {limit < WIDE_LIMIT && total > limit ? (
            <Pressable type="button" className="hist-link" onClick={() => setLimit(WIDE_LIMIT)}>
              {pageText(locale, `Load ${WIDE_LIMIT}`, `טעינת ${WIDE_LIMIT}`)}
            </Pressable>
          ) : null}
        </footer>
      ) : null}
      </div>
    </section>
  );
}
