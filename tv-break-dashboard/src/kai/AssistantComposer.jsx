import React from 'react';
import { Button } from '@mui/material';
import { Bot, Send, Sparkles } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';

// The way in: the empty state that offers a first question, and the composer
// that sends one. Split out of AssistantPanel so both files stay under the
// file-size cap; the state still lives in the panel, so this renders and calls
// back and holds nothing of its own.
//
// The send control becomes a stop control while an answer is in flight, in the
// same place, because the measured failure this console replaced was a browser
// with no reply and nothing to press.

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

export function AssistantComposer({ locale, composerRef, question, onQuestionChange, onKeyDown, unavailable, asking, onSend, onStop }) {
  return (
    <>
      <div className="asst-composer">
        <textarea
          ref={composerRef}
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          onKeyDown={onKeyDown}
          rows={1}
          maxLength={2000}
          dir={question ? 'auto' : (locale === 'he' ? 'rtl' : 'ltr')}
          placeholder={unavailable ? pageText(locale, 'Kai is not available right now', 'קאי אינו זמין כרגע') : pageText(locale, 'Ask about the plan or request a change, in Hebrew or English', 'שאלו על התוכנית או בקשו שינוי, בעברית או באנגלית')}
          disabled={unavailable}
          aria-label={pageText(locale, 'Question for Kai', 'שאלה לקאי')}
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
      <p className="asst-hint">{pageText(locale, 'Enter sends, Shift+Enter adds a line, Cmd+J opens Kai from any screen.', 'מקש Enter שולח, Shift+Enter יורד שורה, Cmd+J פותח את קאי מכל מסך.')}</p>
    </>
  );
}
