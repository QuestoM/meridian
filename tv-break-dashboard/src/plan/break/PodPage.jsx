import React, { useCallback, useEffect, useState } from 'react';
import { pageText } from '../../shell/format';
import PodBoard from './PodBoard';
import { figureText } from './pod-model';
import './break-pod.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

// The break contents surface: every pod a traffic file declares on one day.
//
// The day comes from the files on disk rather than from the plan, because the
// pod is read from the traffic file and only the traffic file names a break per
// advertisement. A day with no file behind it is a state of the data and reads as
// one, with the days that are covered named so a person knows where to go next
// instead of only what is missing.
function PodPage({ locale, notify }) {
  const he = locale === 'he';
  const label = (en, hebrew) => (he ? hebrew : en);
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState('');
  const [day, setDay] = useState('');
  const [openId, setOpenId] = useState('');
  const [busy, setBusy] = useState(false);

  const load = useCallback((wanted) => {
    const query = wanted ? `?day=${encodeURIComponent(wanted)}` : '';
    return fetch(`${API_BASE}/api/breaks/pods${query}`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return response.json();
      })
      .then((body) => {
        setPayload(body);
        setDay(body.day || '');
        setError('');
        return body;
      })
      .catch((fetchError) => { setError(fetchError.message); return null; });
  }, []);

  useEffect(() => { load(''); }, [load]);

  const pods = (payload && Array.isArray(payload.pods) ? payload.pods : []);
  const open = pods.find((pod) => pod.pod_id === openId) || null;

  const write = (method, podId, body) => {
    setBusy(true);
    return fetch(`${API_BASE}/api/breaks/pod/${encodeURIComponent(podId)}/order`, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return response.json();
      })
      .then(() => load(day))
      .catch((writeError) => {
        if (notify) notify(writeError.message, writeError.message);
        else setError(writeError.message);
      })
      .finally(() => setBusy(false));
  };

  const covered = (payload && Array.isArray(payload.covered_days) ? payload.covered_days : []);

  return (
    <section className="pod-page" dir={he ? 'rtl' : 'ltr'}>
      <header className="pod-page-head">
        <h2>{pageText(locale, 'Break contents', 'תוכן הברייק')}</h2>
        <p dir="auto">
          {label(
            'The spots inside each break, read from the daily traffic file, with the arithmetic between what they declare and how long the break runs.',
            'התשדירים בתוך כל ברייק, כפי שנקראו מקובץ הטראפיק היומי, יחד עם החשבון שבין מה שהם מצהירים לבין משך הברייק בפועל.',
          )}
        </p>
      </header>

      {error && <p className="pod-warning" dir="auto">{error}</p>}

      {covered.length > 0 && (
        <div className="pod-days">
          <span dir="auto">{label('Days a traffic file covers', 'ימים שקובץ טראפיק מכסה')}</span>
          {covered.map((covering) => (
            <button
              key={covering}
              type="button"
              className={`pod-day${covering === day ? ' pod-day-open' : ''}`}
              dir="ltr"
              aria-pressed={covering === day}
              onClick={() => { setOpenId(''); load(covering); }}
            >
              {covering}
            </button>
          ))}
        </div>
      )}

      {payload && !payload.available && (
        <div className="pod-empty">
          <p dir="auto">{(he && payload.reason_he) || payload.reason}</p>
          <p dir="auto">{(he && payload.path_forward_he) || payload.path_forward}</p>
        </div>
      )}

      {payload && payload.available && (
        <ol className="pod-list">
          {pods.map((pod) => (
            <li key={pod.pod_id}>
              <button
                type="button"
                className={`pod-row${pod.pod_id === openId ? ' pod-row-open' : ''}`}
                aria-expanded={pod.pod_id === openId}
                onClick={() => setOpenId(pod.pod_id === openId ? '' : pod.pod_id)}
              >
                <span dir="ltr">{pod.break_start_clock}</span>
                <span dir="auto">{pod.programme && pod.programme.value ? pod.programme.value : label('no programme named', 'לא צוינה תוכנית')}</span>
                <span dir="ltr">{pod.arithmetic.spot_count} {label('spots', 'תשדירים')}</span>
                <span dir="ltr">{figureText(pod.arithmetic.declared_load, locale)}</span>
                <span dir="ltr">{figureText(pod.arithmetic.unfilled, locale)} {label('unfilled', 'לא מכוסה')}</span>
                {pod.order && pod.order.state === 'operator' && (
                  <span className="pod-tag" dir="auto">{label('operator order', 'סדר של המפעיל')}</span>
                )}
              </button>
              {pod.pod_id === openId && (
                <PodBoard
                  pod={pod}
                  locale={locale}
                  busy={busy}
                  onSaveOrder={(keys) => write('PUT', pod.pod_id, { spot_keys: keys })}
                  onRevertOrder={() => write('DELETE', pod.pod_id, null)}
                />
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

export default PodPage;
