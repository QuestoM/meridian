import React, { useEffect, useMemo, useRef, useState } from 'react';
import { GripVertical, Lock, Unlock } from 'lucide-react';
import { pageText } from '../../shell/format';
import { Code, Figure, Name } from '../../shell/bidi';
import { formatDay, formatStamp } from '../../shell/dates';
import {
  arithmeticRows,
  continuityRows,
  copyCheckCoverage,
  copyLengthBadge,
  declaredVerdict,
  dropIndexFor,
  elapsedSeconds,
  fieldReason,
  fieldText,
  lockState,
  moveSpot,
  orderChanged,
  orderKeys,
  orderNote,
  positionCode,
  positionLabel,
  positionViolationLabel,
  positionViolationMap,
  secondsLabel,
  spotColumns,
  spotsInOrder,
  verificationList,
} from './pod-model';
import './break-pod.css';

// One break's pod, seen and reordered.
//
// The three figures at the top are the reason this surface exists: what the pod's
// spots declare between them, how long the pod runs from its own declared start,
// and the seconds in between that no spot covers. Each prints its basis beside
// it, because a figure a person cannot check is a figure they have to trust. The
// declared break length is a fourth figure and often absent; when it is, this
// says so in words rather than letting a zero stand in for a length nobody
// declared.
//
// The verification list is the trade's own first step: every copy version that
// disagrees with its booked duration, every spot with no declared length, and
// every spot not airing in the position the traffic file declares. Locking is the
// second step: it freezes whichever order is on screen and refuses a further
// write until an operator unlocks it.
//
// Reorder is a drag and it is also a keyboard. Alt with the up and down arrows
// moves the focused spot one place, because a traffic operator working a pod of
// thirty eight spots should not have to aim, and a drag shows where the spot will
// land before it is dropped. What a reorder cannot do is change the cast: the
// spots come from the traffic file, so the surface says so rather than leaving a
// person hunting for an add control.
//
// readOnly renders the same pod with no reorder surface at all, rather than a
// busy state that leaves the rows draggable and the acts merely disabled with no
// reason. The pod's own page carries the acts and their inverse.
//
// Direction. This section carries no dir of its own: it inherits the app shell's
// own direction root, and every figure, code and name below is isolated with the
// primitives from shell/bidi.jsx rather than with a dir attribute.
function PodBoard({ pod, locale, onSaveOrder, onRevertOrder, onLock, onUnlock, busy, readOnly }) {
  const he = locale === 'he';
  const label = (en, hebrew) => (he ? hebrew : en);
  const [keys, setKeys] = useState(() => orderKeys(pod));
  const [dragging, setDragging] = useState(null);
  const [dropTarget, setDropTarget] = useState(null);
  const listRef = useRef(null);

  // An error names a spot, so it opens it rather than leaving a reader to find
  // it by eye in a pod of thirty eight.
  const openSpot = (spotKey) => {
    const list = listRef.current;
    const row = list && list.querySelector(`[data-spot-key="${spotKey}"]`);
    if (!row) return;
    row.scrollIntoView({ block: 'center' });
    row.focus();
  };

  useEffect(() => { setKeys(orderKeys(pod)); }, [pod]);

  const spots = useMemo(() => spotsInOrder(pod, keys), [pod, keys]);
  const rows = useMemo(() => arithmeticRows(pod, locale), [pod, locale]);
  const continuity = useMemo(() => continuityRows(pod, locale), [pod, locale]);
  const columns = useMemo(() => spotColumns(locale), [locale]);
  const verdict = useMemo(() => declaredVerdict(pod, locale), [pod, locale]);
  // Over the spots as the surface currently shows them, not only as served, so
  // a move shows its consequence before a save spends a write on it.
  const violations = useMemo(() => positionViolationMap(spots), [spots]);
  const errors = useMemo(() => verificationList(spots, locale), [spots, locale]);
  const elapsed = useMemo(() => elapsedSeconds(spots), [spots]);
  const coverage = useMemo(() => copyCheckCoverage(spots), [spots]);
  const lock = useMemo(() => lockState(pod), [pod]);
  const changed = orderChanged(pod, keys);
  const saved = ((pod && pod.order) || {}).state === 'operator';
  const stale = ((pod && pod.order) || {}).state === 'stale';
  // A stale row is still a row, and the act that clears it is the same DELETE
  // either way. Gating the revert act on the applied state left a pod that had
  // just been told its saved order is not applied with no way to clear it.
  const hasSavedOrder = saved || stale;
  const savedAt = ((pod && pod.order) || {}).saved_at || '';
  const missing = Number((pod && pod.arithmetic && pod.arithmetic.spots_missing_a_length) || 0);

  // A locked pod is frozen, so it does not move on screen either. Letting the
  // rows drag while the order is finalised put an order on screen that no
  // record held, under a heading that said Locked.
  const frozen = readOnly || lock.locked;

  const move = (from, to) => setKeys((current) => moveSpot(current, from, to));

  const onKeyDown = (event, index) => {
    if (frozen || !event.altKey) return;
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      move(index, index - 1);
    } else if (event.key === 'ArrowDown') {
      event.preventDefault();
      move(index, index + 1);
    }
  };

  const onRowDragOver = (event, index) => {
    if (frozen || dragging === null) return;
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const edge = event.clientY - rect.top < rect.height / 2 ? 'before' : 'after';
    setDropTarget({ index, edge });
  };

  const onRowDrop = (event, index) => {
    if (frozen) return;
    event.preventDefault();
    const target = dropTarget && dropTarget.index === index ? dropTarget : { index, edge: 'after' };
    move(dragging, dropIndexFor(dragging, target.index, target.edge));
    setDragging(null);
    setDropTarget(null);
  };

  return (
    <section className="pod-board">
      <header className="pod-head">
        <div>
          <h3><Name>{fieldText(pod.programme, locale)}</Name></h3>
          <p className="pod-when">
            <Figure>{pod.break_start_clock}</Figure>
            {pod.break_type && (
              <span title={label('The traffic file\'s own code, not yet a translated vocabulary term', 'הקוד המקורי מקובץ הטראפיק, ללא אוצר מילים מתורגם עדיין')}>
                {label('Break type', 'סוג ברייק')} <Code>{pod.break_type}</Code>
              </span>
            )}
            <span>{label('on', 'בערוץ')} <Name>{pod.channel && pod.channel.value ? pod.channel.value : label('no channel in settings', 'לא הוגדר ערוץ בהגדרות')}</Name></span>
          </p>
          {lock.locked && (
            <p className="pod-lock-tag">
              <Lock size={12} aria-hidden="true" />
              {label('Locked', 'נעול')}
              {lock.lockedBy && <span> {label('by', 'על ידי')} <Name>{lock.lockedBy}</Name></span>}
              {lock.lockedAt && <Figure className="pod-lock-when">{formatStamp(lock.lockedAt) || lock.lockedAt}</Figure>}
              {!lock.lockedBy && <span>{label('by nobody the register could name, because sign-in is not set up yet', 'ללא שם ברשומה, מפני שהכניסה למערכת טרם הוגדרה')}</span>}
            </p>
          )}
        </div>
        <code><Code>{pod.pod_id}</Code></code>
      </header>

      <p className="pod-basis">{pod.channel && ((he && pod.channel.basis_he) || pod.channel.basis)}</p>
      {pod.boundary && (
        <p className="pod-basis">
          {label('One break is every spot sharing', 'ברייק אחד הוא כל תשדיר החולק את')} <Name>{pod.boundary.value}</Name>
          {', '}{(he && pod.boundary.basis_he) || pod.boundary.basis}
        </p>
      )}

      <div className="pod-figures">
        {rows.map((row) => (
          <div key={row.key} className={`pod-figure pod-figure-${row.key}`}>
            <span className="pod-figure-label">{row.label}</span>
            <strong><Figure>{row.text}</Figure></strong>
            <span className="pod-figure-basis">{row.basis}</span>
          </div>
        ))}
        <div className={`pod-figure pod-figure-checks${errors.length > 0 ? ' pod-figure-checks-warning' : ''}`}>
          <span className="pod-figure-label">
            {changed ? label('Verification if saved', 'אימות אם יישמר') : label('Verification', 'אימות')}
          </span>
          <strong><Figure>{errors.length}</Figure></strong>
          <span className="pod-figure-basis">
            {errors.length > 0
              ? label('spots below need a look before this pod airs', 'תשדירים שלמטה דורשים בדיקה לפני השידור')
              : coverage.checked > 0
                ? label(
                    `no disagreement in the length or the position, and the copy check ran on ${coverage.checked} of ${coverage.total} spots. The rest declare no length in their copy version to check against.`,
                    `אין אי-התאמה באורך או במיקום, ובדיקת הגרסה רצה על ${coverage.checked} מתוך ${coverage.total} תשדירים. השאר אינם מצהירים על אורך בשם הגרסה שאפשר לבדוק מולו.`,
                  )
                : label(
                    'no disagreement in the length or the position, and no spot here declares a length in its copy version to check against.',
                    'אין אי-התאמה באורך או במיקום, ואף תשדיר כאן אינו מצהיר על אורך בשם הגרסה שאפשר לבדוק מולו.',
                  )}
            {changed && (
              <>
                {' '}
                {label('computed on the order shown here, before it is saved', 'מחושב על הסדר המוצג כאן, לפני שמירתו')}
              </>
            )}
          </span>
        </div>
      </div>

      {continuity.length > 0 && (
        <p className="pod-continuity">
          <span className="pod-continuity-label">{label('Where the uncovered seconds sit', 'היכן יושבות השניות הלא מכוסות')}</span>
          {continuity.map((row) => (
            <span key={row.key} className="pod-continuity-part">
              {row.count}, <Figure>{row.seconds}</Figure>
            </span>
          ))}
          <span className="pod-figure-basis">
            {label(
              'the dead air before the first spot plus the holes between spots, from the times the traffic file declares. These sit inside the uncovered figure above.',
              'האוויר המת לפני התשדיר הראשון בתוספת הרווחים בין התשדירים, לפי השעות שקובץ הטראפיק מצהיר עליהן. אלה נמצאים בתוך המספר הלא מכוסה שלמעלה.',
            )}
          </span>
        </p>
      )}

      {missing > 0 && (
        <p className="pod-warning">
          {label(
            `${missing} of these spots declare no length, so the sum above is a floor and the difference cannot be computed.`,
            `${missing} מהתשדירים כאן אינם מצהירים על אורך, ולכן הסכום שלמעלה הוא רצפה ולא ניתן לחשב את ההפרש.`,
          )}
        </p>
      )}

      <div className={`pod-verdict pod-verdict-${verdict.state === 'real' ? verdict.verdict : 'unavailable'}`}>
        <strong>{verdict.headline}</strong>
        {verdict.state === 'real' ? (
          <span>
            <Figure>{secondsLabel(verdict.declaredSeconds, locale)}</Figure> {label('declared', 'מוצהר')} / <Figure>{secondsLabel(verdict.loadSeconds, locale)}</Figure> {label('sold', 'נמכר')} / <Figure>{secondsLabel(verdict.seconds, locale)}</Figure>
          </span>
        ) : (
          <span>{verdict.detail}</span>
        )}
        {verdict.state !== 'real' && verdict.covers.length > 0 && (
          <span className="pod-figure-basis">
            {label('The plan on disk covers', 'התוכנית שבדיסק מכסה')}
            {' '}
            <Figure>{formatDay(verdict.covers[0])}</Figure>
            {' '}
            {label('to', 'עד')}
            {' '}
            <Figure>{formatDay(verdict.covers[verdict.covers.length - 1])}</Figure>
          </span>
        )}
      </div>

      {errors.length > 0 && (
        <ul className="pod-errors">
          {errors.map((error) => (
            <li key={error.key} className={`pod-error pod-error-${error.kind}`}>
              <button type="button" className="pod-error-open" onClick={() => openSpot(error.spotKey)}>
                <span className="pod-error-advertiser"><Name>{error.advertiser}</Name></span>
                <span>{error.detail}</span>
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className="pod-order-note">
        <p>{orderNote(pod, locale)}</p>
        {stale && (
          <p className="pod-warning">
            {label(
              'This pod changed after that order was saved, so the order is kept on record and not applied. Clear it with Back to the traffic file order, or move the spots again and save.',
              'התוכן הזה השתנה לאחר שהסדר נשמר, ולכן הסדר נשמר ברשומה ואינו מיושם. אפשר למחוק אותו בעזרת חזרה לסדר קובץ הטראפיק, או להזיז את התשדירים שוב ולשמור.',
            )}
            {savedAt && (
              <span className="pod-figure-basis">
                {label('Saved on', 'נשמר בתאריך')} <Figure>{formatStamp(savedAt) || savedAt}</Figure>
              </span>
            )}
          </p>
        )}
        {lock.locked && (
          <p className="pod-warning">
            {label(
              'This pod is locked. The order below is frozen, and unlocking it is what lets a spot move again.',
              'התוכן הזה נעול. הסדר שלמטה מוקפא, וביטול הנעילה הוא שמאפשר להזיז תשדיר שוב.',
            )}
          </p>
        )}
      </div>

      {readOnly && (
        <p className="pod-hint">
          {label(
            'Reordering and locking happen from Break contents, this pod\'s own page.',
            'סידור מחדש ונעילה מתבצעים מתוך תוכן הברייק, העמוד של התוכן הזה עצמו.',
          )}
        </p>
      )}
      {!frozen && (
        <p className="pod-hint">
          {label(
            'Drag a spot to move it, or hold Alt and press the up or down arrow on a focused spot.',
            'גררו תשדיר כדי להזיז אותו, או החזיקו Alt והקישו על חץ למעלה או למטה על תשדיר במיקוד.',
          )}
          <span className="pod-figure-basis">
            {label(
              'The spots themselves come from the traffic file, so one cannot be added, removed or moved to another break here. Only their order is yours.',
              'התשדירים עצמם מגיעים מקובץ הטראפיק, ולכן לא ניתן להוסיף, להסיר או להעביר תשדיר לברייק אחר כאן. רק הסדר שלהם נתון בידיכם.',
            )}
          </span>
        </p>
      )}

      {!readOnly && changed && !lock.locked && (
        <p className="pod-hint">
          {label(
            'Save this order before locking the pod, so the order that is frozen is the order that is recorded.',
            'שמרו את הסדר הזה לפני נעילת התוכן, כדי שהסדר שמוקפא יהיה הסדר שנרשם.',
          )}
        </p>
      )}

      {(changed || saved) && (
        <p className="pod-figure-basis">
          {label(
            'The start times below are the ones the traffic file declares. They do not move with the order.',
            'שעות ההתחלה שלמטה הן אלה שקובץ הטראפיק מצהיר עליהן. הן אינן זזות עם הסדר.',
          )}
        </p>
      )}

      <div className="pod-spot-table" role="table" aria-label={label('The pod, one row per spot', 'תוכן הברייק, שורה לכל תשדיר')}>
        <div className="pod-spot-head" role="row">
          {columns.map((column) => (
            <span key={column.key} className={`pod-head-${column.key}`} role="columnheader">{column.label}</span>
          ))}
        </div>

        <ol className="pod-spots" role="rowgroup" ref={listRef}>
          {spots.map((spot, index) => {
            const violation = violations.get(spot.spot_key);
            const badge = copyLengthBadge(spot.copy_length, locale);
            const dropClass = !frozen && dropTarget && dropTarget.index === index
              ? ` pod-spot-drop-${dropTarget.edge}`
              : '';
            return (
              <li
                key={spot.spot_key}
                role="row"
                data-spot-key={spot.spot_key}
                className={`pod-spot${frozen ? ' pod-spot-frozen' : ''}${dragging === index ? ' pod-spot-dragging' : ''}${dropClass}`}
                draggable={!frozen}
                tabIndex={0}
                onKeyDown={(event) => onKeyDown(event, index)}
                onDragStart={() => !frozen && setDragging(index)}
                onDragEnd={() => { setDragging(null); setDropTarget(null); }}
                onDragOver={(event) => onRowDragOver(event, index)}
                onDrop={(event) => onRowDrop(event, index)}
                aria-label={`${index + 1}, ${fieldText(spot.advertiser, locale)}, ${positionLabel(spot.position, locale)}, ${spot.duration.state === 'real' ? secondsLabel(spot.duration.seconds, locale) : label('length unknown', 'אורך לא ידוע')}, ${badge.text}${violation ? `, ${positionViolationLabel(violation, locale)}` : ''}`}
              >
                <span className="pod-grip-cell" role="cell" aria-hidden="true">
                  <GripVertical size={14} className="pod-grip" />
                </span>
                <span className="pod-seq" role="cell">
                  <Figure title={label('Order in the pod', 'הסדר בתוך הברייק')}>{index + 1}</Figure>
                </span>
                <span
                  className={`pod-pos pod-pos-${spot.position.kind}${spot.position.preferred ? ' pod-pos-preferred' : ''}${violation ? ' pod-pos-violated' : ''}`}
                  role="cell"
                >
                  <Figure title={violation ? positionViolationLabel(violation, locale) : `${label('Position', 'מיקום')}: ${positionLabel(spot.position, locale)}`}>
                    {positionCode(spot.position)}
                  </Figure>
                </span>
                <span className="pod-advertiser" role="cell">
                  <Name>{fieldText(spot.advertiser, locale)}</Name>
                  {fieldReason(spot.advertiser, locale) && (
                    <em className="pod-missing">{fieldReason(spot.advertiser, locale)}</em>
                  )}
                </span>
                <span className="pod-creative" role="cell"><Name>{fieldText(spot.creative, locale)}</Name></span>
                <span className={`pod-copy pod-copy-${badge.state}`} role="cell" title={badge.detail || ''}>{badge.text}</span>
                <span className="pod-house" role="cell"><Code>{fieldText(spot.house_number, locale)}</Code></span>
                <span className="pod-type" role="cell" title={label('The traffic file\'s own code, not yet a translated vocabulary term', 'הקוד המקורי מקובץ הטראפיק, ללא אוצר מילים מתורגם עדיין')}>
                  <Code>{spot.spot_type}</Code>
                </span>
                <span className="pod-clock" role="cell"><Figure>{spot.start_clock || '?'}</Figure></span>
                <span className="pod-elapsed" role="cell" title={label('Seconds already aired in this break by the time this spot starts, in the order shown here', 'שניות ששודרו כבר בברייק הזה עד תחילת התשדיר הזה, לפי הסדר המוצג כאן')}>
                  {elapsed[index] === null || elapsed[index] === undefined
                    ? label('unknown', 'לא ידוע')
                    : <Figure>{secondsLabel(elapsed[index], locale)}</Figure>}
                </span>
                <span className={`pod-len${spot.duration.state === 'real' ? '' : ' pod-missing'}`} role="cell">
                  {spot.duration.state === 'real' ? <Figure>{secondsLabel(spot.duration.seconds, locale)}</Figure> : label('length unknown', 'אורך לא ידוע')}
                </span>
              </li>
            );
          })}
        </ol>
      </div>

      {!readOnly && (
        <div className="pod-acts">
          <button
            type="button"
            className="pod-act pod-act-save"
            disabled={!changed || busy || lock.locked}
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
            disabled={!hasSavedOrder || busy || lock.locked}
            onClick={() => onRevertOrder()}
          >
            {pageText(locale, 'Back to the traffic file order', 'חזרה לסדר קובץ הטראפיק')}
          </button>
          {lock.locked ? (
            <button type="button" className="pod-act pod-act-unlock" disabled={busy || !onUnlock} onClick={() => onUnlock()}>
              <Unlock size={13} aria-hidden="true" />
              {pageText(locale, 'Unlock', 'ביטול נעילה')}
            </button>
          ) : (
            <button type="button" className="pod-act pod-act-lock" disabled={busy || changed || !onLock} onClick={() => onLock()}>
              <Lock size={13} aria-hidden="true" />
              {pageText(locale, 'Lock this pod', 'נעילת התוכן')}
            </button>
          )}
        </div>
      )}
    </section>
  );
}

export default PodBoard;
