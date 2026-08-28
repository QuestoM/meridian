import React from 'react';
import { Button } from '../studio/actions';
import { Bot, Send, Sparkles } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { MentionPicker } from './MentionPicker';
import { useMentions } from './mention-state';
import { AttachedRefs, MentionOverlay } from './MentionRefs';
import { liveRefs } from './mention-refs';
import { Pressable, TextAreaControl } from '../studio/dom-controls';

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
      <p>{pageText(locale, 'No questions asked yet. Mabat answers from saved data only, and the conversation is saved for your return.', 'עוד לא נשאלו שאלות. מבט עונה רק מן הנתונים השמורים, והשיחה נשמרת לחזרה הבאה.')}</p>
      {showSuggestions ? (
        <div className="asst-suggestions">
          <span className="asst-suggestions-label"><Sparkles size={12} />{pageText(locale, 'You can start with one of these', 'אפשר להתחיל מאחת מאלה')}</span>
          {SUGGESTIONS.map((pair) => (
            <Pressable type="button" className="asst-suggestion" key={pair[1]} onClick={() => onPick(pageText(locale, pair[0], pair[1]))}>
              {pageText(locale, pair[0], pair[1])}
            </Pressable>
          ))}
        </div>
      ) : null}
    </div>
  );
}

// THE @ TRIGGER IS WIRED HERE, and the panel is untouched by it.
//
// The panel holds the question and the send key handler, so the picker's own
// state lives outside both (mention-state.js) and the panel's onKeyDown is
// CALLED THROUGH rather than replaced: while the picker is open that hook
// answers the arrow keys, Enter and Escape, and every other keystroke falls
// through to the panel exactly as it did before. Enter with no picker open still
// sends. Nothing is taken away from anyone.
//
// The caret is read after the browser has applied the keystroke, which is why
// the run is recomputed from the textarea rather than from the previous value.
//
// WHY THE TEXT CHANGE GOES THROUGH THE HOOK. A reference is a span into this
// exact string, so every edit has to carry the spans across it or drop the ones
// it ran through. That arithmetic is mention-refs.js's and it is applied on the
// one path every keystroke, paste and undo already takes. The prop the panel
// passes is bound below under the name the line already used, so the keystroke
// path itself is unchanged: the same call still tells the panel the text moved.

// ``attachments`` rides INSIDE the composer row rather than on a row of its
// own above it. In the dock the panel is a narrow column and every band of
// chrome is taken from the conversation; the paperclip belongs beside the
// field it attaches to anyway.
export function AssistantComposer({ locale, composerRef, question, onQuestionChange: setText, refs, onRefsChange, onKeyDown, unavailable, asking, onSend, onStop, onActivity, attachments = null }) {
  const activity = onActivity || (() => {});
  const overlayRef = React.useRef(null);
  const mention = useMentions({
    locale,
    composerRef,
    question,
    onQuestionChange: setText,
    refs: refs || [],
    onRefsChange: onRefsChange || (() => {}),
  });
  const onQuestionChange = mention.applyText;
  // What is STILL attached to this exact sentence. A span survives only while
  // the characters it covers are the label it was made from, so the strip and
  // the highlight show what would actually be sent and never a stale binding.
  const attached = liveRefs(question, refs || []);
  return (
    <>
      <div className="asst-composer">
        {/* The field is one positioned box holding two layers that must line up
            to the pixel: the textarea, which is what sizes the box and what the
            operator types into, and the highlight pinned to the same box behind
            it. Their metrics are stated once each and checked against each other
            by test, because a highlight a pixel off its words is worse than no
            highlight at all. The overlay carries no glyph of its own for the same
            reason: a character here that is not in the field would slide it. */}
        <div className="mention-field">
          <MentionOverlay ref={overlayRef} text={question} refs={attached} />
          {/* The trigger reads the TEXT and the caret rather than the keystroke, so
              a paste and an undo are seen exactly as typing is. onInput is a second
              listener on the same native event, deliberately: onChange below stays
              the line the keep-warm test pins character for character, and the
              question-being-written signal it carries is not touched by any of this. */}
          <TextAreaControl
            ref={composerRef}
            value={question}
            onFocus={() => activity()}
            onChange={(event) => { activity(); onQuestionChange(event.target.value); }}
            onInput={(event) => mention.sync(event.target)}
            onKeyUp={(event) => mention.sync(event.target)}
            onClick={(event) => mention.sync(event.target)}
            onBlur={() => mention.sync(null)}
            onKeyDown={(event) => mention.onKeyDown(event, onKeyDown)}
            // A grown question scrolls inside its own box, so the layer under it
            // has to scroll with it or the highlight parts company with the words
            // the moment the field passes its maximum height.
            onScroll={(event) => { if (overlayRef.current) overlayRef.current.scrollTop = event.target.scrollTop; }}
            rows={1}
            maxLength={2000}
            dir={question ? 'auto' : (locale === 'he' ? 'rtl' : 'ltr')}
            placeholder={unavailable ? pageText(locale, 'Mabat is not available right now', 'מבט אינו זמין כרגע') : pageText(locale, 'Ask about the plan or request a change, in Hebrew or English', 'שאלו על התוכנית או בקשו שינוי, בעברית או באנגלית')}
            disabled={unavailable}
            aria-label={pageText(locale, 'Question for Mabat', 'שאלה למבט')}
            aria-expanded={mention.open}
            aria-controls={mention.open ? 'kai-mention-picker' : undefined}
          />
        </div>
        <MentionPicker
          locale={locale}
          anchorEl={composerRef ? composerRef.current : null}
          open={mention.open && !unavailable}
          rows={mention.rows}
          absent={mention.absent}
          omitted={mention.omitted}
          loading={mention.loading}
          activeIndex={mention.active}
          trail={mention.trail}
          onChoose={mention.choose}
          onHover={mention.setActive}
          onDescend={mention.descend}
          onUpTo={mention.upTo}
        />
        {attachments}
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
      {/* What is attached to this sentence right now, before anything is sent.
          The binding being invisible is the exact defect this piece was built
          against, so it is a thing on screen and not a thing only the model
          reads. What each reference came back AS appears under the finished
          question, because resolution happens at send time and never before. */}
      <AttachedRefs locale={locale} refs={attached} />
      {/* @ is offered, never required: every question that worked before still
          works typed out in full, which is the whole design of this trigger. */}
      <p className="asst-hint">{pageText(locale, 'Enter sends, Shift+Enter adds a line, @ points at a broadcast day, a programme, an advertiser, an agency or a calendar event, the arrow key on the reading edge goes inside one, Cmd+J opens Mabat from any screen.', 'מקש Enter שולח, Shift+Enter יורד שורה, @ מצביע על יום שידור, תוכנית, מפרסם, סוכנות או אירוע לוח שנה, מקש החץ בקצה הקריאה נכנס פנימה, Cmd+J פותח את מבט מכל מסך.')}</p>
    </>
  );
}
