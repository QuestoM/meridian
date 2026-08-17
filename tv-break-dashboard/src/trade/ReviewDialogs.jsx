import React, { useMemo, useRef, useState } from 'react';
import { Dialog } from '../studio/modal';
import { Button } from '../studio/actions';
import { InputControl, SelectControl, TextAreaControl } from '../studio/dom-controls';
import { Status } from '../studio';
import { TriangleAlert } from 'lucide-react';
import { Code, Name, Prose } from '../shell/bidi';
import { pageText } from '../shell/format';
import { familyName, termName, termsByFamily } from './trade-terms';

// The five acts a reviewer can perform that need more than a press.
//
// WHY THE FIELDS ARE EDITED AS THE STORED OBJECT. Sixty-three terms carry
// sixty-three different parameter schemas — a CPP table by daypart, a tier
// ladder, a percentage with a stated base, a list of forbidden programmes. A
// generated form per schema would be a second, drifting copy of the taxonomy on
// this side of the wire, and the first schema it got wrong would silently write a
// term the compiler then refuses. So the reviewer edits the object itself, the
// text is parsed before it is sent, and the server's own schema validation is the
// authority: an unknown key or a bad shape comes back as the store's refusal and
// is printed verbatim.
//
// A REJECTION NEEDS A REASON and the server enforces it, because a term removed
// silently is a term dropped. The dialog refuses to submit without one rather
// than letting the operator discover the rule from a 422.
//
// A REVIEWER-ADDED TERM NEEDS PROVENANCE: either a clause id with a quote that
// appears verbatim in that clause, or an explicit declaration that the term was
// agreed outside the document. Both are offered; neither is assumed.

function parseParams(text) {
  const trimmed = String(text || '').trim();
  if (!trimmed) return { ok: true, value: {} };
  try {
    const value = JSON.parse(trimmed);
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return { ok: false, error: 'object' };
    }
    return { ok: true, value };
  } catch (error) {
    return { ok: false, error: 'syntax' };
  }
}

function paramsHint(locale, state) {
  if (state === 'object') {
    return pageText(
      locale,
      'The fields have to be a JSON object: a set of named values.',
      'השדות צריכים להיות אובייקט JSON: אוסף של ערכים בשמות.',
    );
  }
  return pageText(
    locale,
    'That is not valid JSON, so it was not sent. Nothing was changed.',
    'זה אינו JSON תקין, ולכן לא נשלח. דבר לא שונה.',
  );
}

function Field({ label, hint, children }) {
  return (
    <label className="trd-field">
      <span className="trd-field-label">{label}</span>
      {children}
      {hint ? <span className="trd-field-hint">{hint}</span> : null}
    </label>
  );
}

export default function ReviewDialogs({ action, locale, busy, error, onClose, onSubmit }) {
  const [reason, setReason] = useState('');
  const [note, setNote] = useState('');
  const [text, setText] = useState(() => (
    action && action.term ? JSON.stringify(action.term.editedParams || action.term.params || {}, null, 1) : '{}'
  ));
  const [termId, setTermId] = useState('');
  const [clauseId, setClauseId] = useState(() => (action && action.clause ? action.clause.clause_id : ''));
  const [quote, setQuote] = useState('');
  const [notInDocument, setNotInDocument] = useState(false);
  const [winner, setWinner] = useState(() => (
    action && action.conflict && action.conflict.instances.length > 0 ? action.conflict.instances[0] : ''
  ));
  const [parseState, setParseState] = useState('');
  const initialFocus = useRef(null);
  const groups = useMemo(() => termsByFamily(), []);
  if (!action) return null;

  const kind = action.kind;

  function submitReject(event) {
    event.preventDefault();
    if (!reason.trim()) return;
    onSubmit({ kind, verdict: 'rejected', reason: reason.trim() });
  }

  function submitEdit(event) {
    event.preventDefault();
    const parsed = parseParams(text);
    if (!parsed.ok) { setParseState(parsed.error); return; }
    setParseState('');
    onSubmit({ kind, verdict: 'edited', edited_params: parsed.value, reason: reason.trim() });
  }

  function submitAdd(event) {
    event.preventDefault();
    const parsed = parseParams(text);
    if (!parsed.ok) { setParseState(parsed.error); return; }
    if (!termId) return;
    if (!notInDocument && (!clauseId.trim() || !quote.trim())) return;
    setParseState('');
    onSubmit({
      kind,
      term_id: termId,
      params: parsed.value,
      clause_id: notInDocument ? null : clauseId.trim(),
      quote: notInDocument ? '' : quote.trim(),
      not_in_document: notInDocument,
      note: note.trim(),
    });
  }

  function submitAcknowledge(event) {
    event.preventDefault();
    if (!note.trim()) return;
    onSubmit({ kind, note: note.trim() });
  }

  function submitConflict(event) {
    event.preventDefault();
    if (!winner) return;
    onSubmit({ kind, winner_instance_id: winner, note: note.trim() });
  }

  const titles = {
    reject: pageText(locale, 'Reject this term', 'דחיית המונח'),
    edit: pageText(locale, 'Correct the extracted values', 'תיקון הערכים שחולצו'),
    add: pageText(locale, 'Add a term the reading missed', 'הוספת מונח שהקריאה החמיצה'),
    acknowledge: pageText(locale, 'Acknowledge a clause with no term', 'אישור סעיף שאין לו מונח'),
    conflict: pageText(locale, 'Decide which clause governs', 'הכרעה איזה סעיף קובע'),
  };

  const descriptions = {
    reject: pageText(
      locale,
      'The term stays on the record as rejected, with your reason, and is excluded from the approved version.',
      'המונח יישאר ברשומה כמונח שנדחה, עם הנימוק שלכם, ולא ייכלל בגרסה המאושרת.',
    ),
    edit: pageText(
      locale,
      'Your values are stored beside the document\'s own; both stay visible and the approved version carries yours.',
      'הערכים שלכם נשמרים לצד אלה של המסמך; שניהם נשארים גלויים והגרסה המאושרת נושאת את שלכם.',
    ),
    add: pageText(
      locale,
      'A term the reading did not propose. It needs either the clause and its exact words, or a statement that it was agreed outside the document.',
      'מונח שהקריאה לא הציעה. נדרשים או הסעיף ולשונו המדויקת, או הצהרה שהוא הוסכם מחוץ למסמך.',
    ),
    acknowledge: pageText(
      locale,
      'A clause the reading could not map to any term. Acknowledging it says a person read it and owns it; the note becomes part of the approved record.',
      'סעיף שהקריאה לא הצליחה למפות לשום מונח. אישור ידני אומר שאדם קרא אותו ולוקח עליו אחריות; ההערה תיכנס לרשומה המאושרת.',
    ),
    conflict: pageText(
      locale,
      'Two clauses of this agreement say different things. Choose the one that governs.',
      'שני סעיפים בהסכם הזה אומרים דברים שונים. בחרו את זה שקובע.',
    ),
  };

  const forms = {
    reject: submitReject,
    edit: submitEdit,
    add: submitAdd,
    acknowledge: submitAcknowledge,
    conflict: submitConflict,
  };

  const canSubmit = {
    reject: Boolean(reason.trim()),
    edit: true,
    add: Boolean(termId) && (notInDocument || (Boolean(clauseId.trim()) && Boolean(quote.trim()))),
    acknowledge: Boolean(note.trim()),
    conflict: Boolean(winner),
  }[kind];

  return (
    <Dialog
      open
      onClose={onClose}
      size="wide"
      title={titles[kind]}
      description={descriptions[kind]}
      closeLabel={pageText(locale, 'Close', 'סגירה')}
      initialFocusRef={initialFocus}
      className="trd-action-dialog"
      footer={(
        <>
          <Button type="button" variant="outlined" onClick={onClose}>
            {pageText(locale, 'Cancel', 'ביטול')}
          </Button>
          <Button type="submit" form="trd-action-form" disabled={!canSubmit || busy}>
            {busy
              ? pageText(locale, 'Saving', 'שומר')
              : pageText(locale, 'Save', 'שמירה')}
          </Button>
        </>
      )}
    >
      <form id="trd-action-form" className="trd-form" onSubmit={forms[kind]}>
        {action.term ? (
          <p className="trd-dialog-subject">
            <Name>{termName(action.term.term_id, locale)}</Name>
            <Code>{action.term.instance_id}</Code>
          </p>
        ) : null}

        {action.clause ? (
          <div className="trd-dialog-subject-clause">
            <Code>{action.clause.clause_id}</Code>
            <Prose>{action.clause.text}</Prose>
          </div>
        ) : null}

        {kind === 'reject' ? (
          <Field
            label={pageText(locale, 'Why this term is rejected', 'מדוע המונח נדחה')}
            hint={pageText(locale, 'Required. A term removed without a reason is a term dropped.', 'שדה חובה. מונח שהוסר בלי נימוק הוא מונח שנעלם.')}
          >
            <TextAreaControl
              ref={initialFocus}
              rows={3}
              value={reason}
              required
              onChange={(event) => setReason(event.target.value)}
            />
          </Field>
        ) : null}

        {kind === 'add' ? (
          <Field label={pageText(locale, 'Which term', 'איזה מונח')}>
            <SelectControl ref={initialFocus} value={termId} onChange={(event) => setTermId(event.target.value)}>
              <option value="">{pageText(locale, 'Choose a term', 'בחרו מונח')}</option>
              {groups.map((group) => (
                <optgroup key={group.family} label={familyName(group.family, locale)}>
                  {group.terms.map((entry) => (
                    <option key={entry.id} value={entry.id}>
                      {locale === 'he' ? entry.he : entry.en}
                    </option>
                  ))}
                </optgroup>
              ))}
            </SelectControl>
          </Field>
        ) : null}

        {kind === 'add' ? (
          <label className="trd-checkbox">
            <InputControl
              type="checkbox"
              checked={notInDocument}
              onChange={(event) => setNotInDocument(event.target.checked)}
            />
            <span>
              {pageText(
                locale,
                'This term is not in the document; it was agreed elsewhere.',
                'המונח אינו במסמך; הוא הוסכם במקום אחר.',
              )}
            </span>
          </label>
        ) : null}

        {kind === 'add' && !notInDocument ? (
          <>
            <Field
              label={pageText(locale, 'The clause it comes from', 'הסעיף שממנו הוא בא')}
              hint={pageText(locale, 'The clause id as it appears in the clause list.', 'מזהה הסעיף כפי שהוא מופיע ברשימת הסעיפים.')}
            >
              <InputControl type="text" value={clauseId} onChange={(event) => setClauseId(event.target.value)} />
            </Field>
            <Field
              label={pageText(locale, 'Its exact words', 'לשונו המדויקת')}
              hint={pageText(
                locale,
                'The quote has to appear verbatim in that clause. The server checks it and refuses a paraphrase.',
                'הציטוט חייב להופיע במדויק בסעיף. השרת בודק זאת ומסרב לניסוח מחדש.',
              )}
            >
              <TextAreaControl rows={2} value={quote} onChange={(event) => setQuote(event.target.value)} />
            </Field>
          </>
        ) : null}

        {kind === 'edit' || kind === 'add' ? (
          <Field
            label={pageText(locale, 'The fields', 'השדות')}
            hint={pageText(
              locale,
              'The term\'s own parameters, as the engine stores them. The server validates them against the term\'s schema and refuses a field the term does not take.',
              'הפרמטרים של המונח, כפי שהמנוע שומר אותם. השרת מאמת אותם מול הסכימה של המונח ומסרב לשדה שהמונח אינו מקבל.',
            )}
          >
            <TextAreaControl
              ref={kind === 'edit' ? initialFocus : undefined}
              className="trd-json"
              rows={12}
              value={text}
              onChange={(event) => { setText(event.target.value); setParseState(''); }}
            />
          </Field>
        ) : null}

        {parseState ? (
          <p className="trd-form-error" role="alert">
            <TriangleAlert size={14} aria-hidden="true" />
            {paramsHint(locale, parseState)}
          </p>
        ) : null}

        {kind === 'conflict' ? (
          <Field label={pageText(locale, 'The term that governs', 'המונח הקובע')}>
            <SelectControl ref={initialFocus} value={winner} onChange={(event) => setWinner(event.target.value)}>
              {action.conflict.instances.map((instanceId) => (
                <option key={instanceId} value={instanceId}>{instanceId}</option>
              ))}
            </SelectControl>
          </Field>
        ) : null}

        {kind === 'conflict' && action.conflict.contested ? (
          <Prose className="trd-dialog-body">{action.conflict.contested}</Prose>
        ) : null}

        {kind === 'conflict' && action.conflict.rule ? (
          <p className="trd-field-hint">
            {pageText(locale, 'The resolver proposed:', 'מנגנון ההכרעה הציע:')}
            {' '}
            <Prose as="span">{action.conflict.rule}</Prose>
          </p>
        ) : null}

        {kind === 'acknowledge' || kind === 'conflict' || kind === 'add' ? (
          <Field
            label={pageText(locale, 'Note', 'הערה')}
            hint={kind === 'acknowledge'
              ? pageText(locale, 'Required. Say what the clause is and who owns it.', 'שדה חובה. כתבו מהו הסעיף ומי אחראי עליו.')
              : pageText(locale, 'Optional.', 'לא חובה.')}
          >
            <TextAreaControl
              ref={kind === 'acknowledge' ? initialFocus : undefined}
              rows={3}
              value={note}
              required={kind === 'acknowledge'}
              onChange={(event) => setNote(event.target.value)}
            />
          </Field>
        ) : null}

        {error ? (
          <p className="trd-form-error" role="alert">
            <TriangleAlert size={14} aria-hidden="true" />
            <Status status="danger">{pageText(locale, 'The server refused this', 'השרת סירב לפעולה')}</Status>
            <Prose as="span">{error}</Prose>
          </p>
        ) : null}
      </form>
    </Dialog>
  );
}
