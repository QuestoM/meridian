import React, { useEffect, useState } from 'react';
import { Button, FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import { Activity, RefreshCcw } from 'lucide-react';
import { API_BASE } from '../shell/api';
import { SIGN_IN_LABELS, pair } from './history-labels';
import { actLabel, outcomeOf } from './history-refused';

// Compact humanized label for an activity entry. The classification is the
// server's: GET /api/activity-log carries the same action code History reads,
// so one vocabulary names an act in both places and no surface has to match on
// an HTTP path. An entry with no known action falls back to a method and path
// code chip, exactly as this panel has always done.
//
// It also carries the outcome, and that closes here the same defect History had:
// the status was in the payload and this panel printed the word for the act
// beside it, so a refused write read as one that happened. This log holds its
// fields at the top level rather than under facts, which outcomeOf accepts.
export function activityActionLabel(entry, he) {
  const locale = he ? 'he' : 'en';
  const event = entry.event || '';
  if (event && event !== 'request') return pair(SIGN_IN_LABELS, event, locale) || null;
  const action = entry.action || '';
  if (!action || action === 'other') return null;
  return actLabel(action, outcomeOf(entry), locale) || null;
}

export function activityTimeLabel(ts, he) {
  if (!ts) return '';
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString(he ? 'he-IL' : 'en-US', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
}

// The settings-page activity log: who changed what and when, served by
// GET /api/activity-log. The API decides visibility (admin sees everyone, any
// other role only itself, dev mode without login sees everything), so this
// panel renders the scope it was given and says so honestly instead of
// pretending to filter anything client-side.
export function ActivityLogPanel({ locale }) {
  const he = locale === 'he';
  const [log, setLog] = useState({ status: 'loading', entries: [], scope: 'all' });
  const [userFilter, setUserFilter] = useState('');
  const [knownUsers, setKnownUsers] = useState([]);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLog((current) => ({ ...current, status: 'loading' }));
    (async () => {
      try {
        const filter = userFilter ? `&user=${encodeURIComponent(userFilter)}` : '';
        const response = await fetch(`${API_BASE}/api/activity-log?limit=100${filter}`, { credentials: 'include' });
        if (!response.ok) throw new Error(`${response.status}`);
        const data = await response.json();
        if (cancelled) return;
        const entries = Array.isArray(data.entries) ? data.entries : [];
        setLog({ status: 'ready', entries, scope: data.scope === 'self' ? 'self' : 'all' });
        if (!userFilter) {
          setKnownUsers((current) => {
            const merged = new Set(current);
            entries.forEach((entry) => {
              if (entry.user) merged.add(entry.user);
            });
            return Array.from(merged).sort();
          });
        }
      } catch {
        if (!cancelled) setLog({ status: 'error', entries: [], scope: 'all' });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [userFilter, reloadKey]);

  const showUserColumn = log.scope === 'all';
  const filterLabel = he ? 'סינון לפי מפעיל' : 'Filter by operator';
  return (
    <section className="settings-panel wide">
      <div className="settings-panel-head">
        <div>
          <h2>{he ? 'יומן פעילות' : 'Activity log'}</h2>
          <p>{he ? 'מי שינה מה ומתי, כולל פעולות שבוצעו דרך עוזר ה־AI' : 'Who changed what and when, including actions made through the AI assistant'}</p>
        </div>
        <Activity size={18} />
      </div>
      <div className="alog-toolbar">
        <div className="alog-controls">
          {showUserColumn && knownUsers.length > 0 && (
            <FormControl size="small" className="alog-filter">
              {/* The default is the empty value (all operators): displayEmpty +
                  renderValue make it read as a real selection, and the label is
                  pinned shrunk so it never overlaps the rendered text. */}
              <InputLabel id="alog-user-filter" shrink>{filterLabel}</InputLabel>
              <Select
                labelId="alog-user-filter"
                label={filterLabel}
                value={userFilter}
                displayEmpty
                renderValue={(selected) => (selected ? selected : (he ? 'כל המפעילים' : 'All operators'))}
                onChange={(event) => setUserFilter(event.target.value)}
              >
                <MenuItem value="">{he ? 'כל המפעילים' : 'All operators'}</MenuItem>
                {knownUsers.map((name) => (
                  <MenuItem key={name} value={name}>{name}</MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
          {log.scope === 'self' && (
            <span className="alog-self-note">{he ? 'מוצגת הפעילות שלכם בלבד' : 'Showing your own activity only'}</span>
          )}
          <a className="alog-self-note" href="#Versions">
            {he ? 'לצפייה בהיסטוריה המלאה, כולל הרצות ונקודות שחזור' : 'Open the full history, including runs and restore points'}
          </a>
        </div>
        <Button
          type="button"
          variant="outlined"
          className="run-button"
          disabled={log.status === 'loading'}
          onClick={() => setReloadKey((key) => key + 1)}
        >
          <RefreshCcw size={15} />
          {he ? 'רענון' : 'Refresh'}
        </Button>
      </div>
      {log.status === 'loading' && <p className="alog-note">{he ? 'רק רגע...' : 'Loading...'}</p>}
      {log.status === 'error' && (
        <p className="alog-note alog-error" role="alert">{he ? 'טעינת יומן הפעילות נכשלה. אפשר לנסות לרענן.' : 'Could not load the activity log. Try refreshing.'}</p>
      )}
      {log.status === 'ready' && log.entries.length === 0 && (
        <p className="alog-note">
          {userFilter
            ? (he ? 'אין רשומות למפעיל שנבחר.' : 'No entries for the selected operator.')
            : (he ? 'אין עדיין רשומות ביומן. פעולות שינוי יופיעו כאן.' : 'No activity recorded yet. Changes will appear here.')}
        </p>
      )}
      {log.status === 'ready' && log.entries.length > 0 && (
        <div className="alog-table-wrap">
          <table className="alog-table">
            <thead>
              <tr>
                <th>{he ? 'זמן' : 'Time'}</th>
                {showUserColumn && <th>{he ? 'מפעיל' : 'Operator'}</th>}
                <th>{he ? 'פעולה' : 'Action'}</th>
                <th>{he ? 'סטטוס' : 'Status'}</th>
              </tr>
            </thead>
            <tbody>
              {log.entries.map((entry, index) => {
                const label = activityActionLabel(entry, he);
                const status = Number(entry.status);
                const hasStatus = Number.isFinite(status) && status > 0;
                return (
                  <tr key={`${entry.ts || 'entry'}-${index}`}>
                    <td><span className="alog-time" dir="ltr">{activityTimeLabel(entry.ts, he)}</span></td>
                    {showUserColumn && <td><span className="alog-user">{entry.user || ''}</span></td>}
                    <td>
                      {label ? <span>{label}</span> : <code className="alog-code" dir="ltr">{`${entry.method || ''} ${entry.path || ''}`.trim()}</code>}
                      {entry.via === 'assistant' && <span className="alog-via">{he ? 'עוזר AI' : 'AI assistant'}</span>}
                    </td>
                    <td>{hasStatus ? <span className={`alog-status${status >= 400 ? ' warn' : ''}`} dir="ltr">{status}</span> : null}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default ActivityLogPanel;
