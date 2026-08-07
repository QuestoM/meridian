import React, { useEffect, useMemo, useState } from 'react';
import { GripVertical } from 'lucide-react';
import { pageText } from '../../shell/format';
import {
  arithmeticRows,
  declaredVerdict,
  fieldReason,
  fieldText,
  moveSpot,
  orderChanged,
  orderKeys,
  orderNote,
  positionCode,
  positionLabel,
  secondsLabel,
  spotsInOrder,
} from './pod-model';
import './break-pod.css';

// One break's pod, seen and reordered.
//
// The three figures at the top are the reason this surface exists: what the pod's
// spots declare between them, how long the pod actually runs from its own
// declared start, and the seconds in between that no spot covers. Each prints the
// basis it was computed on beside it, because a figure a person cannot check is a
// figure they have to trust.
//
// The declared break length is a fourth figure and it is often absent. When it is,
// this says so in words and leaves the arithmetic against it blank, rather than
// letting a zero stand in for a length nobody declared.
//
// Reorder is a drag and it is also a keyboard. Alt with the up and down arrows
// moves the focused spot one place, because a traffic operator working a pod of
// thirty eight spots should not have to aim.
function PodBoard({ pod, locale, onSaveOrder, onRevertOrder, busy }) {
  const he = locale === 'he';
  const label = (en, hebrew) => (he ? hebrew : en);
  const [keys, setKeys] = useState(() => orderKeys(pod));
  const [dragging, setDragging] = useState(null);

  useEffect(() => { setKeys(orderKeys(pod)); }, [pod]);

  const spots = useMemo(() => spotsInOrder(pod, keys), [pod, keys]);
  const rows = useMemo(() => arithmeticRows(pod, locale), [pod, locale]);
  const verdict = useMemo(() => declaredVerdict(pod, locale), [pod, locale]);
  const changed = orderChanged(pod, keys);
  const saved = ((pod && pod.order) || {}).state === 'operator';
  const stale = ((pod && pod.order) || {}).state === 'stale';
  const missing = Number((pod && pod.arithmetic && pod.arithmetic.spots_missing_a_length) || 0);

  const move = (from, to) => setKeys((current) => moveSpot(current, from, to));

  const onKeyDown = (event, index) => {
    if (!event.altKey) return;
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      move(index, index - 1);
    } else if (event.key === 'ArrowDown') {
      event.preventDefault();
      move(index, index + 1);
    }
  };

  return (
    <section className="pod-board" dir={he ? 'rtl' : 'ltr'}>
      <header className="pod-head">
        <div>
          <h3 dir="auto">{fieldText(pod.programme, locale)}</h3>
          <p className="pod-when">
            <span dir="ltr">{pod.break_start_clock}</span>
            <span dir="auto">{pod.break_type}</span>
            <span dir="auto">{label('on', 'בערוץ')} {pod.channel && pod.channel.value ? pod.channel.value : label('no channel in settings', 'לא הוגדר ערוץ בהגדרות')}</span>
          </p>
        </div>
        <code dir="ltr">{pod.pod_id}</code>
      </header>

      <p className="pod-basis" dir="auto">{pod.channel && ((he && pod.channel.basis_he) || pod.channel.basis)}</p>

      <div className="pod-figures">
        {rows.map((row) => (
          <div key={row.key} className={`pod-figure pod-figure-${row.key}`}>
            <span className="pod-figure-label" dir="auto">{row.label}</span>
            <strong dir="ltr">{row.text}</strong>
            <span className="pod-figure-basis" dir="auto">{row.basis}</span>
          </div>
        ))}
      </div>

      {missing > 0 && (
        <p className="pod-warning" dir="auto">
          {label(
            `${missing} of these spots declare no length, so the sum above is a floor and the difference cannot be computed.`,
            `${missing} מהתשדירים כאן אינם מצהירים על אורך, ולכן הסכום שלמעלה הוא רצפה ולא ניתן לחשב את ההפרש.`,
          )}
        </p>
      )}

      <div className={`pod-verdict pod-verdict-${verdict.state === 'real' ? verdict.verdict : 'unavailable'}`}>
        <strong dir="auto">{verdict.headline}</strong>
        {verdict.state === 'real' ? (
          <span dir="ltr">
            {secondsLabel(verdict.declaredSeconds, locale)} {label('declared', 'מוצהר')} / {secondsLabel(verdict.loadSeconds, locale)} {label('sold', 'נמכר')} / {secondsLabel(verdict.seconds, locale)}
          </span>
        ) : (
          <span dir="auto">{verdict.detail}</span>
        )}
      </div>

      <div className="pod-order-note">
        <p dir="auto">{orderNote(pod, locale)}</p>
        {stale && (
          <p className="pod-warning" dir="auto">
            {label(
              'The saved order names spots this pod no longer holds, so it is not applied.',
              'הסדר השמור נוקב בתשדירים שהתוכן הזה כבר אינו מכיל, ולכן הוא אינו מיושם.',
            )}
          </p>
        )}
      </div>

      <p className="pod-hint" dir="auto">
        {label(
          'Drag a spot to move it, or hold Alt and press the up or down arrow on a focused spot.',
          'גררו תשדיר כדי להזיז אותו, או החזיקו Alt והקישו על חץ למעלה או למטה על תשדיר במיקוד.',
        )}
      </p>

      <ol className="pod-spots">
        {spots.map((spot, index) => (
          <li
            key={spot.spot_key}
            className={`pod-spot${dragging === index ? ' pod-spot-dragging' : ''}`}
            draggable
            tabIndex={0}
            onKeyDown={(event) => onKeyDown(event, index)}
            onDragStart={() => setDragging(index)}
            onDragEnd={() => setDragging(null)}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => { event.preventDefault(); move(dragging, index); setDragging(null); }}
            aria-label={`${index + 1}, ${fieldText(spot.advertiser, locale)}, ${positionLabel(spot.position, locale)}, ${spot.duration.state === 'real' ? secondsLabel(spot.duration.seconds, locale) : label('length unknown', 'אורך לא ידוע')}`}
          >
            <GripVertical size={14} className="pod-grip" aria-hidden="true" />
            <span className="pod-seq" dir="ltr">{index + 1}</span>
            <span
              className={`pod-pos pod-pos-${spot.position.kind}${spot.position.preferred ? ' pod-pos-preferred' : ''}`}
              dir="ltr"
              title={positionLabel(spot.position, locale)}
            >
              {positionCode(spot.position)}
            </span>
            <span className="pod-advertiser" dir="auto">
              {fieldText(spot.advertiser, locale)}
              {fieldReason(spot.advertiser, locale) && (
                <em className="pod-missing" dir="auto">{fieldReason(spot.advertiser, locale)}</em>
              )}
            </span>
            <span className="pod-creative" dir="auto">{fieldText(spot.creative, locale)}</span>
            <span className="pod-house" dir="ltr">{fieldText(spot.house_number, locale)}</span>
            <span className="pod-type" dir="auto">{spot.spot_type}</span>
            <span className="pod-clock" dir="ltr">{spot.start_clock || '?'}</span>
            <span className={`pod-len${spot.duration.state === 'real' ? '' : ' pod-missing'}`} dir={spot.duration.state === 'real' ? 'ltr' : 'auto'}>
              {spot.duration.state === 'real' ? secondsLabel(spot.duration.seconds, locale) : label('length unknown', 'אורך לא ידוע')}
            </span>
          </li>
        ))}
      </ol>

      <div className="pod-acts">
        <button
          type="button"
          className="pod-act pod-act-save"
          disabled={!changed || busy}
          onClick={() => onSaveOrder(keys)}
        >
          {pageText(locale, 'Save this order', 'שמירת הסדר הזה')}
        </button>
        <button
          type="button"
          className="pod-act"
          disabled={!changed || busy}
          onClick={() => setKeys(orderKeys(pod))}
        >
          {pageText(locale, 'Discard the change', 'ביטול השינוי')}
        </button>
        <button
          type="button"
          className="pod-act"
          disabled={!saved || busy}
          onClick={() => onRevertOrder()}
        >
          {pageText(locale, 'Back to the traffic file order', 'חזרה לסדר קובץ הטראפיק')}
        </button>
      </div>
    </section>
  );
}

export default PodBoard;
