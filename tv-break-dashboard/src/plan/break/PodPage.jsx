import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Pressable } from '../../studio/dom-controls';
import { pageText } from '../../shell/format';
import { Figure, Name, Prose } from '../../shell/bidi';
import { formatDay, todayIso } from '../../shell/dates';
import PodBoard from './PodBoard';
import PreferredPositionRate from './PreferredPositionRate';
import { countLabel, figureText, podAttentionScore } from './pod-model';
import './break-pod.css';
import './break-pod-list.css';
import '../day/master-control-broadcast.css';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

// Today, and every day this page prints, come from shell/dates.js. Reading the
// calendar parts off a Date here answered in the machine's own zone rather than
// the broadcast one, which moves the answer by a day for a reader west of
// Israel, and it printed the ISO form to a person, which is the machine's
// format and not the one an Israeli operator reads.

function readParam(name) {
  if (typeof window === 'undefined') return '';
  return new URLSearchParams(window.location.search).get(name) || '';
}

// Keeps this page's day and open pod in the address bar, without touching the
// workspace hash a click here did not change. A bookmark or a paste of the URL
// reopens the same pod, and the covered-days line elsewhere in the product can
// link straight to it.
function writeParams(day, podId, method = 'replaceState') {
  if (typeof window === 'undefined') return;
  const params = new URLSearchParams(window.location.search);
  if (day) params.set('day', day); else params.delete('day');
  if (podId) params.set('pod', podId); else params.delete('pod');
  const query = params.toString();
  window.history[method]({ ...(window.history.state || {}), day, pod: podId }, '', `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`);
}

// The break contents surface: every pod a traffic file declares on one day.
//
// The day comes from the files on disk rather than from the plan, because the
// pod is read from the traffic file and only the traffic file names a break per
// advertisement. A day with no file behind it is a state of the data and reads as
// one, with the days that are covered named so a person knows where to go next
// instead of only what is missing.
//
// The traffic operator's own word for this job is tonight's breaks. Today's own
// calendar date opens first when a traffic file covers it; every day the shipped
// data actually covers is 2025-04-27, so on any other day this says so plainly
// rather than pretending tonight has contents it does not.
//
// requestedDay is the state channel a covered-day link elsewhere on this same
// page uses to open a day here. A hash assignment is a no-op when the hash
// already names this page, which is exactly the case a click inside this
// page's own break drawer is in, so a caller already on this page hands this
// component a fresh {day, token} instead and this effect answers it directly.
function PodPage({ locale, notify, requestedDay }) {
  const he = locale === 'he';
  const label = (en, hebrew) => (he ? hebrew : en);
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState('');
  const [day, setDay] = useState('');
  const [openId, setOpenId] = useState(() => readParam('pod'));
  const [busy, setBusy] = useState(false);
  const [sortByAttention, setSortByAttention] = useState(false);
  const sectionRef = useRef(null);

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

  useEffect(() => {
    const wanted = readParam('day') || todayIso();
    load(wanted).then((body) => {
      // Today's date named nothing a traffic file covers, so the fallback is
      // the first covered day rather than a blank page.
      if (body && !body.available && !readParam('day')) {
        load('').then((fallback) => {
          if (fallback) writeParams(fallback.day || '', readParam('pod'), 'replaceState');
        });
      } else if (body) {
        writeParams(body.day || wanted, readParam('pod'), 'replaceState');
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load]);

  useEffect(() => {
    function restoreAddress() {
      const addressedDay = readParam('day') || todayIso();
      setOpenId(readParam('pod'));
      load(addressedDay);
    }
    window.addEventListener('popstate', restoreAddress);
    return () => window.removeEventListener('popstate', restoreAddress);
  }, [load]);

  // Opening a pod low in the day list put its arithmetic below the fold and
  // scrolled nothing, so the three figures the whole surface exists for were off
  // screen at the moment they were asked for. Measured at 1440 x 960 on the ninth
  // of ten rows: 65 px past the viewport bottom. The row itself goes to the top
  // rather than the figures, so the reader keeps the pod they clicked in view.
  useEffect(() => {
    if (!openId || typeof document === 'undefined') return;
    const node = document.querySelector(`[data-pod="${openId}"]`);
    if (node) node.scrollIntoView({ block: 'start' });
  }, [openId]);

  useEffect(() => {
    if (!requestedDay || !requestedDay.day) return;
    setOpenId('');
    writeParams(requestedDay.day, '', 'replaceState');
    load(requestedDay.day);
    if (sectionRef.current) sectionRef.current.scrollIntoView({ block: 'start' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestedDay]);

  const pods = (payload && Array.isArray(payload.pods) ? payload.pods : []);
  const sorted = useMemo(() => {
    if (!sortByAttention) return pods;
    return pods.slice().sort((a, b) => podAttentionScore(b) - podAttentionScore(a));
  }, [pods, sortByAttention]);
  const open = pods.find((pod) => pod.pod_id === openId) || null;

  const write = (method, podId, path, body) => {
    setBusy(true);
    return fetch(`${API_BASE}/api/breaks/pod/${encodeURIComponent(podId)}/${path}`, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    })
      .then(async (response) => {
        if (!response.ok) {
          // The server already wrote the sentence a refused write should show,
          // as the response body's own detail: locked, already locked, not
          // locked, or a key list that is not this pod's own. A raw status
          // line is only what is left when that body cannot be read at all.
          const detail = await response.json().catch(() => null);
          throw new Error((detail && detail.detail) || `${response.status} ${response.statusText}`);
        }
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
  const tonight = todayIso();
  const tonightCovered = covered.includes(tonight);

  return (
    <section className="pod-page broadcast-pods" ref={sectionRef}>
      <header className="pod-page-head">
        <h1>{pageText(locale, "Break contents, tonight's breaks", 'תוכן הברייק, הברייקים של הערב')}</h1>
        <p>
          {label(
            'The spots inside each break, read from the daily traffic file, with the arithmetic between what they declare and how long the break runs.',
            'התשדירים בתוך כל ברייק, כפי שנקראו מקובץ הטראפיק היומי, יחד עם החשבון שבין מה שהם מצהירים לבין משך הברייק בפועל.',
          )}
        </p>
      </header>

      {error && <Prose as="p" className="pod-warning">{error}</Prose>}

      {covered.length > 0 && !tonightCovered && (
        <p className="pod-tonight">
          <Figure>{formatDay(tonight)}</Figure>
          {' '}
          {label(
            'has no traffic file, so tonight\'s own breaks are not on disk. Showing the days that are covered instead.',
            'ללא קובץ טראפיק, ולכן הברייקים של הערב עצמו אינם בדיסק. מוצגים במקום זאת הימים שיש להם כיסוי.',
          )}
        </p>
      )}

      {covered.length > 0 && (
        <div className="pod-days">
          <span>{label('Days a traffic file covers', 'ימים שקובץ טראפיק מכסה')}</span>
          {covered.map((covering) => (
            <Pressable
              key={covering}
              type="button"
              className={`pod-day${covering === day ? ' pod-day-open' : ''}`}
              aria-pressed={covering === day}
              onClick={() => {
                setOpenId('');
                writeParams(covering, '', 'pushState');
                load(covering);
              }}
            >
              <Figure>{formatDay(covering)}</Figure>
            </Pressable>
          ))}
        </div>
      )}

      {payload && !payload.available && (
        <div className="pod-empty">
          <p>{(he && payload.reason_he) || payload.reason}</p>
          <p>{(he && payload.path_forward_he) || payload.path_forward}</p>
        </div>
      )}

      {payload && payload.available && (
        <>
          <div className="pod-list-tools">
            <Pressable
              type="button"
              className={`pod-sort-act${!sortByAttention ? ' pod-sort-act-active' : ''}`}
              aria-pressed={!sortByAttention}
              onClick={() => setSortByAttention(false)}
            >
              {label('Time order', 'סדר שידור')}
            </Pressable>
            <Pressable
              type="button"
              className={`pod-sort-act${sortByAttention ? ' pod-sort-act-active' : ''}`}
              aria-pressed={sortByAttention}
              onClick={() => setSortByAttention(true)}
            >
              {label('Needs attention first', 'דורש תשומת לב קודם')}
            </Pressable>
          </div>
          <ol className="pod-list">
            {sorted.map((pod) => {
              const attention = Number(pod.verification && pod.verification.count) || 0;
              return (
                <li key={pod.pod_id} data-pod={pod.pod_id}>
                  <Pressable
                    type="button"
                    className={`pod-row${pod.pod_id === openId ? ' pod-row-open' : ''}${attention > 0 ? ' pod-row-attention' : ''}`}
                    aria-expanded={pod.pod_id === openId}
                    onClick={() => {
                      const next = pod.pod_id === openId ? '' : pod.pod_id;
                      setOpenId(next);
                      writeParams(day, next, 'pushState');
                    }}
                  >
                    <Figure>{pod.break_start_clock}</Figure>
                    <span><Name>{pod.programme && pod.programme.value ? pod.programme.value : label('no programme named', 'לא צוינה תוכנית')}</Name></span>
                    <span>{countLabel(pod.arithmetic.spot_count, locale, 'spot', 'spots', 'תשדיר', 'תשדירים')}</span>
                    <span><Figure>{figureText(pod.arithmetic.declared_load, locale)}</Figure> {label('sold', 'נמכר')}</span>
                    <span><Figure>{figureText(pod.arithmetic.unfilled, locale)}</Figure> {label('unfilled', 'לא מכוסה')}</span>
                    {attention > 0 && (
                      <span className="pod-tag pod-tag-attention">
                        {label(`${attention} to check`, `${attention} לבדיקה`)}
                      </span>
                    )}
                    {pod.order && pod.order.state === 'operator' && (
                      <span className="pod-tag">{label('operator order', 'סדר של המפעיל')}</span>
                    )}
                    {pod.order && pod.order.locked && (
                      <span className="pod-tag">{label('locked', 'נעול')}</span>
                    )}
                  </Pressable>
                  {pod.pod_id === openId && (
                    <PodBoard
                      pod={pod}
                      locale={locale}
                      busy={busy}
                      onSaveOrder={(keys) => write('PUT', pod.pod_id, 'order', { spot_keys: keys })}
                      onRevertOrder={() => write('DELETE', pod.pod_id, 'order')}
                      onLock={() => write('PUT', pod.pod_id, 'lock')}
                      onUnlock={() => write('DELETE', pod.pod_id, 'lock')}
                    />
                  )}
                </li>
              );
            })}
          </ol>
          <PreferredPositionRate day={day} locale={locale} />
        </>
      )}
    </section>
  );
}

export default PodPage;
