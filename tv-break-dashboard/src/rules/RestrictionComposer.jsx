import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@mui/material';
import { Check, Loader2, Search } from 'lucide-react';
import { pageText } from '../shell/format';
import DateField from '../shell/DateField';
import AiringNights from './AiringNights';
import RestrictionEffect from './RestrictionEffect';
import {
  KINDS,
  buildWhere,
  fetchAirings,
  fetchTitles,
  kindMeta,
  minutes,
  previewRestriction,
  saveRestriction,
} from './rules-lib';

// The composer is one sentence with typed slots, not a form. A representative
// says a thing about a programme; the slots are the words in that thing that
// change. The effect of the draft is fetched the moment the draft is sayable, so
// the cost is already on screen by the time anybody reaches the save.
const PREVIEW_DEBOUNCE_MS = 350;

function Slot({ children, label }) {
  return (
    <span className="rules-slot" aria-label={label}>
      {children}
    </span>
  );
}

export default function RestrictionComposer({ locale, onSaved, notify }) {
  const he = locale === 'he';
  const [kind, setKind] = useState('clean_tail');
  const [params, setParams] = useState({ protected_minutes: 8 });
  const [query, setQuery] = useState('');
  const [title, setTitle] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [airings, setAirings] = useState(null);
  const [day, setDay] = useState('');
  // Both ends of the life of a rule. The store and the engine have always
  // carried a start date, and the composer only offered the end, so a rule that
  // should begin later could not be written and the person writing it had to
  // remember to come back on the day. The server validates the pair.
  const [startsOn, setStartsOn] = useState('');
  const [expiresOn, setExpiresOn] = useState('');
  const [author, setAuthor] = useState('');
  const [reason, setReason] = useState('');
  const [preview, setPreview] = useState(null);
  const [previewing, setPreviewing] = useState(false);
  const [previewError, setPreviewError] = useState('');
  const [saving, setSaving] = useState(false);
  const abortRef = useRef(null);

  useEffect(() => {
    let active = true;
    fetchTitles(query)
      .then((body) => { if (active) setSuggestions(body.titles || []); })
      .catch(() => { if (active) setSuggestions([]); });
    return () => { active = false; };
  }, [query]);

  const draft = useMemo(() => ({
    kind,
    params,
    where: buildWhere({ title, day }),
    starts_on: startsOn,
    expires_on: expiresOn,
    author,
    reason,
  }), [kind, params, title, day, startsOn, expiresOn, author, reason]);

  const sayable = Boolean(title);

  useEffect(() => {
    if (!sayable) {
      setPreview(null);
      setPreviewError('');
      return undefined;
    }
    const timer = setTimeout(() => {
      if (abortRef.current) abortRef.current.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setPreviewing(true);
      setPreviewError('');
      previewRestriction(draft, controller.signal)
        .then((body) => { setPreview(body); setPreviewing(false); })
        .catch((error) => {
          if (error.name === 'AbortError') return;
          setPreview(null);
          setPreviewError(error.message);
          setPreviewing(false);
        });
    }, PREVIEW_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [draft, sayable]);

  const pickTitle = useCallback((next) => {
    setTitle(next);
    setQuery(next);
    setDay('');
    setAirings(null);
    fetchAirings(next).then(setAirings).catch(() => setAirings(null));
  }, []);

  async function save() {
    setSaving(true);
    try {
      const saved = await saveRestriction(draft);
      notify?.(
        'Restriction saved. It applies on the next run of the plan.',
        'ההגבלה נשמרה. היא תחול בהרצה הבאה של התוכנית.',
      );
      setTitle('');
      setQuery('');
      setDay('');
      setPreview(null);
      onSaved?.(saved);
    } catch (error) {
      notify?.(`Saving the restriction failed (${error.message}).`, `שמירת ההגבלה נכשלה (${error.message}).`);
    } finally {
      setSaving(false);
    }
  }

  const meta = kindMeta(kind);
  // A rule is savable when it compiles to something the store can hold. It used
  // to be gated on the per-airing change list, which four of the six kinds never
  // fill because they compile to one scope-level row, so the save was
  // permanently disabled for them while the same draft posted straight to the
  // API returned 201.
  const rows = Number(preview?.compiled_rows || 0);
  const canSave = rows > 0 && !previewing && !saving;
  const bindsNothing = Boolean(preview) && rows > 0 && Number(preview.bound_airings || 0) === 0;

  return (
    <section className="rules-card rules-composer">
      <h2>{pageText(locale, 'Write a restriction', 'כתיבת הגבלה')}</h2>
      <p className="rules-card-lead">
        {pageText(
          locale,
          'Say it the way you would say it out loud. The cost is on screen before you save.',
          'נסחו את זה כמו שהייתם אומרים בקול. העלות מוצגת לפני השמירה.',
        )}
      </p>

      <p className="rules-sentence" dir={he ? 'rtl' : 'ltr'}>
        <Slot label={pageText(locale, 'What the restriction does', 'מה ההגבלה עושה')}>
          <select value={kind} onChange={(event) => {
            const next = event.target.value;
            setKind(next);
            setParams({ ...kindMeta(next).defaults });
          }}>
            {KINDS.map((entry) => (
              <option key={entry.id} value={entry.id}>{he ? entry.he : entry.en}</option>
            ))}
          </select>
        </Slot>
        {meta.param === 'protected_minutes' && (
          <Slot label={pageText(locale, 'Minutes protected', 'דקות מוגנות')}>
            <input
              type="number" min="1" max="120" dir="ltr"
              value={params.protected_minutes ?? 8}
              onChange={(event) => setParams({ protected_minutes: Number(event.target.value) })}
            />
            <span className="rules-slot-unit">{pageText(locale, 'minutes', 'דקות')}</span>
          </Slot>
        )}
        {meta.param === 'count' && (
          <Slot label={pageText(locale, 'Number of breaks', 'מספר ברייקים')}>
            <input
              type="number" min="0" max="20" dir="ltr"
              value={params.count ?? 1}
              onChange={(event) => setParams({ count: Number(event.target.value) })}
            />
            <span className="rules-slot-unit">{pageText(locale, 'breaks', 'ברייקים')}</span>
          </Slot>
        )}
        {meta.param === 'offset_seconds' && (
          <Slot label={pageText(locale, 'Minute into the programme', 'דקה בתוך התוכנית')}>
            <input
              type="number" min="0" max="240" dir="ltr"
              value={minutes(params.offset_seconds ?? 1320)}
              onChange={(event) => setParams({ offset_seconds: Number(event.target.value) * 60 })}
            />
            <span className="rules-slot-unit">{pageText(locale, 'minutes in', 'דקות מההתחלה')}</span>
          </Slot>
        )}
        <span className="rules-sentence-word">{pageText(locale, 'of', 'של')}</span>
        <Slot label={pageText(locale, 'Programme', 'תוכנית')}>
          <span className="rules-typeahead">
            <Search size={13} aria-hidden="true" />
            <input
              type="text"
              value={query}
              placeholder={pageText(locale, 'Start typing a programme', 'התחילו להקליד שם תוכנית')}
              onChange={(event) => { setQuery(event.target.value); setTitle(''); setAirings(null); }}
              dir="auto"
            />
          </span>
        </Slot>
      </p>

      {!title && query.length > 0 && suggestions.length > 0 && (
        <ul className="rules-suggestions">
          {suggestions.slice(0, 8).map((row) => (
            <li key={row.title}>
              <button type="button" onClick={() => pickTitle(row.title)}>
                <span dir="auto">{row.title}</span>
                <small>
                  {pageText(
                    locale,
                    `${row.airings} airings, ${row.planned_breaks} breaks planned`,
                    `${row.airings} שידורים, ${row.planned_breaks} ברייקים בתוכנית`,
                  )}
                </small>
              </button>
            </li>
          ))}
        </ul>
      )}

      {title && airings && (
        <AiringNights
          locale={locale}
          airings={airings.count}
          nights={airings.nights}
          day={day}
          onPick={setDay}
        />
      )}

      <div className="rules-attribution">
        <label>
          <span>{pageText(locale, 'Who is asking', 'מי מבקש')}</span>
          <input type="text" value={author} onChange={(event) => setAuthor(event.target.value)} dir="auto" />
        </label>
        <label>
          <span>{pageText(locale, 'Why', 'סיבה')}</span>
          <input type="text" value={reason} onChange={(event) => setReason(event.target.value)} dir="auto" />
        </label>
        <label>
          <span>{pageText(locale, 'Starts applying on', 'מתחיל לחול בתאריך')}</span>
          <DateField value={startsOn} onChange={setStartsOn} />
        </label>
        <label>
          <span>{pageText(locale, 'Stops applying on', 'מפסיק לחול בתאריך')}</span>
          <DateField value={expiresOn} onChange={setExpiresOn} />
        </label>
      </div>

      <RestrictionEffect
        locale={locale}
        preview={preview}
        previewing={previewing}
        error={previewError}
        sayable={sayable}
      />

      <div className="rules-composer-actions">
        <Button
          className="run-button"
          type="button"
          variant="contained"
          disabled={!canSave}
          onClick={save}
        >
          {saving ? <Loader2 size={14} className="rules-spin" /> : <Check size={14} />}
          {pageText(locale, 'Save the restriction', 'שמירת ההגבלה')}
        </Button>
        {preview && rows === 0 && (
          <span className="rules-inline-note">
            {pageText(
              locale,
              'Nothing in the plan window breaks this rule, so there is nothing to save yet.',
              'שום דבר בחלון התוכנית אינו מפר את הכלל הזה, ולכן אין מה לשמור בשלב זה.',
            )}
          </span>
        )}
        {bindsNothing && (
          <span className="rules-inline-note">
            {pageText(
              locale,
              `This rule can be saved, but the plan engine binds none of the ${preview.matched_airings} airings it matches, so it moves nothing until one of them carries a break it can act on.`,
              `אפשר לשמור את הכלל הזה, אבל מנוע התוכנית אינו מחיל אותו על אף אחד מ-${preview.matched_airings} השידורים שהוא תואם, ולכן הוא אינו משנה דבר עד שאחד מהם יישא ברייק שהכלל יכול לפעול עליו.`,
            )}
          </span>
        )}
      </div>
    </section>
  );
}
