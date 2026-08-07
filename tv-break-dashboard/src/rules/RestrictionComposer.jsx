import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@mui/material';
import { Check, Loader2, Search } from 'lucide-react';
import { pageText } from '../shell/format';
import DateField from '../shell/DateField';
import AiringNights from './AiringNights';
import ProgrammeMatches from './ProgrammeMatches';
import RestrictionEffect from './RestrictionEffect';
import WiderScopeNote from './WiderScopeNote';
import {
  KINDS,
  buildWhere,
  detailWords,
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
  // How many programmes the query actually matched, which is larger than the
  // list the route serves, so the picker can say what it is not showing.
  const [matchCount, setMatchCount] = useState(0);
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
  const [wider, setWider] = useState(null);
  const [previewing, setPreviewing] = useState(false);
  const [previewError, setPreviewError] = useState('');
  const [saving, setSaving] = useState(false);
  const abortRef = useRef(null);

  useEffect(() => {
    let active = true;
    fetchTitles(query)
      .then((body) => {
        if (!active) return;
        setSuggestions(body.titles || []);
        setMatchCount(Number(body.match_count) || 0);
      })
      .catch(() => { if (active) { setSuggestions([]); setMatchCount(0); } });
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
          setPreviewError(detailWords(error, locale));
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
      notify?.(`Saving the restriction failed (${detailWords(error, 'en')}).`, `שמירת ההגבלה נכשלה (${detailWords(error, 'he')}).`);
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
  // A window rule the chosen night does not breach compiles to nothing, so the
  // save has nothing to write. Rather than leave the button shut with no way
  // on, the same sentence is priced again with the night dropped, and the note
  // below offers the wider rule when the run as a whole does breach it. Only
  // this dead-end state fetches, so the ordinary path costs nothing extra.
  const deadEnd = Boolean(preview) && rows === 0 && Boolean(day) && preview.per_airing === true;
  useEffect(() => {
    if (!deadEnd) {
      setWider(null);
      return undefined;
    }
    let alive = true;
    previewRestriction({ ...draft, where: buildWhere({ title }) })
      .then((body) => { if (alive) setWider(body); })
      .catch(() => { if (alive) setWider(null); });
    return () => { alive = false; };
  }, [deadEnd, draft, title]);

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

      <p className="rules-sentence">
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
              type="number" min="1" max="120"
              value={params.protected_minutes ?? 8}
              onChange={(event) => setParams({ protected_minutes: Number(event.target.value) })}
            />
            <span className="rules-slot-unit">{pageText(locale, 'minutes', 'דקות')}</span>
          </Slot>
        )}
        {meta.param === 'count' && (
          <Slot label={pageText(locale, 'Number of breaks', 'מספר ברייקים')}>
            <input
              type="number" min="0" max="20"
              value={params.count ?? 1}
              onChange={(event) => setParams({ count: Number(event.target.value) })}
            />
            <span className="rules-slot-unit">{pageText(locale, 'breaks', 'ברייקים')}</span>
          </Slot>
        )}
        {meta.param === 'offset_seconds' && (
          <Slot label={pageText(locale, 'Minute into the programme', 'דקה בתוך התוכנית')}>
            <input
              type="number" min="0" max="240"
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
            />
          </span>
        </Slot>
      </p>

      {!title && query.length > 0 && suggestions.length > 0 && (
        <ProgrammeMatches
          locale={locale}
          titles={suggestions}
          matchCount={matchCount}
          onPick={pickTitle}
        />
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
          <input type="text" value={author} onChange={(event) => setAuthor(event.target.value)} />
        </label>
        <label>
          <span>{pageText(locale, 'Why', 'סיבה')}</span>
          <input type="text" value={reason} onChange={(event) => setReason(event.target.value)} />
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
          <WiderScopeNote
            locale={locale}
            night={day}
            wider={wider}
            onWiden={() => setDay('')}
          />
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
