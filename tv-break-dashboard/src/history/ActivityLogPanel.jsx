import React, { useEffect, useState } from 'react';
import { Button, FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import { Activity, RefreshCcw } from 'lucide-react';
import { API_BASE } from '../shell/api';

// Compact humanized label for an activity entry. Known actions get a plain
// language name; everything else falls back to a method+path code chip.
export function activityActionLabel(entry, he) {
  const event = entry.event || '';
  if (event === 'login') return he ? 'כניסה למערכת' : 'Signed in';
  if (event === 'login_failed') return he ? 'ניסיון כניסה שנכשל' : 'Failed sign-in attempt';
  if (event === 'logout') return he ? 'יציאה מהמערכת' : 'Signed out';
  const method = entry.method || '';
  const path = entry.path || '';
  if (method === 'PUT' && path === '/api/settings') return he ? 'עדכון הגדרות' : 'Settings update';
  if (method === 'POST' && /^\/api\/assistant\/proposals\/[^/]+\/apply$/.test(path)) return he ? 'אישור הצעות העוזר' : 'Assistant proposal approval';
  if (method === 'POST' && (path === '/api/recompute-schedule' || path === '/api/jobs/recompute')) return he ? 'חישוב מחדש' : 'Recompute';
  return null;
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
