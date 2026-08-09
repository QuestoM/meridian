import React from 'react';
import { Button } from '@mui/material';
import { Bot, Send, Sparkles } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { MentionPicker, useMentionSearch } from './MentionPicker';
import { insertMention, readMentionQuery } from './mention-trigger';

// The way in: the empty state that offers a first question, and the composer
// that sends one. Split out of AssistantPanel so both files stay under the
// file-size cap; the state still lives in the panel, so this renders and calls
// back and holds nothing of its own.
//
// The send control becomes a stop control while an answer is in flight, in the
// same place, because the measured failure this console replaced was a browser
// with no reply and nothing to press.
//
// onActivity fires when a question starts being written, on focus and on each
// keystroke. It is the one thing this file knows that nothing else can know:
// the panel cannot see a cursor land in the box. The panel throttles it
// (kai-keep-warm.js), so calling it per keystroke is what it is for.

export const SUGGESTIONS = [
  ['What is the weekly net and why', 'מה הנטו השבועי ולמה'],
  ['Suggest a way to raise the net without hurting retention', 'הצע דרך להעלות את הנטו בלי לפגוע בשימור'],
  ['Create a restriction that blocks a break in the first 15 minutes of the evening news', 'צרו הגבלה שאין ברייק ב-15 הדקות הראשונות של מהדורת הערב'],
  ['Raise the revenue weight to 65 and run the plan', 'העלה את משקל ההכנסות ל-65 והריצו את התוכנית'],
  ['Get me to a higher net without dropping retention below 0.75', 'הבא אותי לנטו גבוה יותר בלי לרדת מתחת ל-0.75 שימור'],
  ['Suggest settings that raise the weekly net, and show me the effect before I approve', 'הצע הגדרות שמגדילות את הנטו השבועי, ותראה לי את ההשפעה לפני שאאשר'],
];

export function AssistantEmptyThread({ locale, showSuggestions, onPick }) {
  return (
    <div className="asst-thread-empty">
      <Bot size={18} />
      <p>{pageText(locale, 'No questions asked yet. Kai answers from the saved data only, and the conversation is saved and will appear here next time.', 'עוד לא נשאלו שאלות. קאי עונה מהנתונים השמורים בלבד, והשיחה נשמרת ותופיע כאן בפעם הבאה.')}</p>
      {showSuggestions ? (
        <div className="asst-suggestions">
          <span className="asst-suggestions-label"><Sparkles size={12} />{pageText(locale, 'You can start with one of these', 'אפשר להתחיל מאחת מאלה')}</span>
          {SUGGESTIONS.map((pair) => (
            <button type="button" className="asst-suggestion" key={pair[1]} onClick={() => onPick(pageText(locale, pair[0], pair[1]))}>
              {pageText(locale, pair[0], pair[1])}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

// THE @ TRIGGER LIVES HERE, and the panel is untouched by it.
//
// The panel holds the question and the send key handler and is two lines under
// the 450-line law, so the mention state lives in this file and the panel's own
// onKeyDown is CALLED THROUGH rather than replaced: while the picker is open
// this file answers the arrow keys, Enter and Escape, and every other keystroke
// falls through to the panel exactly as it did before. Enter with no picker open
// still sends. Nothing is taken away from anyone.
//
// The caret is read after the browser has applied the keystroke, which is why
// the run is recomputed from the textarea rather than from the previous value.
function useMentions(composerRef, question, onQuestionChange) {
  const [run, setRun] = React.useState(null);
  const [active, setActive] = React.useState(0);
  const open = run !== null;
  const { rows, loading } = useMentionSearch(run ? run.query : '', open);

  React.useEffect(() => { setActive(0); }, [run && run.query]);

  function sync(element) {
    setRun(element ? readMentionQuery(element.value, element.selectionStart) : null);
  }

  function choose(row) {
    if (!run) return;
    const next = insertMention(question, run, row.label);
    setRun(null);
    onQuestionChange(next.text);
    const element = composerRef && composerRef.current;
    if (element) {
      // After React has written the new value, put the caret past the name.
      window.requestAnimationFrame(() => {
        element.focus();
        element.setSelectionRange(next.caret, next.caret);
      });
    }
  }

  function onKeyDown(event, fallback) {
    if (open && event.key === 'Escape') { event.preventDefault(); setRun(null); return; }
    if (open && rows.length) {
      if (event.key === 'ArrowDown') { event.preventDefault(); setActive((i) => (i + 1) % rows.length); return; }
      if (event.key === 'ArrowUp') { event.preventDefault(); setActive((i) => (i - 1 + rows.length) % rows.length); return; }
      if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); choose(rows[active]); return; }
      if (event.key === 'Tab') { event.preventDefault(); choose(rows[active]); return; }
    }
    if (fallback) fallback(event);
  }

  return { open, rows, loading, active, setActive, choose, onKeyDown, sync };
}

export function AssistantComposer({ locale, composerRef, question, onQuestionChange, onKeyDown, unavailable, asking, onSend, onStop, onActivity }) {
  const activity = onActivity || (() => {});
  const mention = useMentions(composerRef, question, onQuestionChange);
  return (
    <>
      <div className="asst-composer">
        {/* The trigger reads the TEXT and the caret rather than the keystroke, so
            a paste and an undo are seen exactly as typing is. onInput is a second
            listener on the same native event, deliberately: onChange below stays
            the line the keep-warm test pins character for character, and the
            question-being-written signal it carries is not touched by any of this. */}
        <textarea
          ref={composerRef}
          value={question}
          onFocus={() => activity()}
          onChange={(event) => { activity(); onQuestionChange(event.target.value); }}
          onInput={(event) => mention.sync(event.target)}
          onKeyUp={(event) => mention.sync(event.target)}
          onClick={(event) => mention.sync(event.target)}
          onBlur={() => mention.sync(null)}
          onKeyDown={(event) => mention.onKeyDown(event, onKeyDown)}
          rows={1}
          maxLength={2000}
          dir={question ? 'auto' : (locale === 'he' ? 'rtl' : 'ltr')}
          placeholder={unavailable ? pageText(locale, 'Kai is not available right now', 'קאי אינו זמין כרגע') : pageText(locale, 'Ask about the plan or request a change, in Hebrew or English', 'שאלו על התוכנית או בקשו שינוי, בעברית או באנגלית')}
          disabled={unavailable}
          aria-label={pageText(locale, 'Question for Kai', 'שאלה לקאי')}
          aria-expanded={mention.open}
          aria-controls={mention.open ? 'kai-mention-picker' : undefined}
        />
        <MentionPicker
          locale={locale}
          anchorEl={composerRef ? composerRef.current : null}
          open={mention.open && !unavailable}
          rows={mention.rows}
          loading={mention.loading}
          activeIndex={mention.active}
          onChoose={mention.choose}
          onHover={mention.setActive}
        />
        {asking ? (
          <Button variant="outlined" size="small" className="asst-send-btn" onClick={onStop}>
            {pageText(locale, 'Stop', 'עצירה')}
          </Button>
        ) : (
          <Button variant="contained" size="small" className="asst-send-btn" onClick={onSend} disabled={unavailable || !question.trim()} endIcon={<Send size={14} style={locale === 'he' ? { transform: 'scaleX(-1)' } : undefined} />}>
            {pageText(locale, 'Send', 'שליחה')}
          </Button>
        )}
      </div>
      {/* @ is offered, never required: every question that worked before still
          works typed out in full, which is the whole design of this trigger. */}
      <p className="asst-hint">{pageText(locale, 'Enter sends, Shift+Enter adds a line, @ points at an advertiser, an agency, a programme or a calendar event, Cmd+J opens Kai from any screen.', 'מקש Enter שולח, Shift+Enter יורד שורה, @ מצביע על מפרסם, סוכנות, תוכנית או אירוע לוח שנה, Cmd+J פותח את קאי מכל מסך.')}</p>
    </>
  );
}
