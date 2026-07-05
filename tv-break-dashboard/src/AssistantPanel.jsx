import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '@mui/material';
import { Bot, Send } from 'lucide-react';
import { API_BASE, pageText } from './surface-helpers';
import './assistant-panel.css';

// AssistantPanel: the in-product AI assistant. The operator asks questions in
// Hebrew or English about the saved schedule and plan; the server composes a
// compact context from the real saved payloads and Claude answers ONLY from
// that context. Every answer is footed with the grounding sources the server
// actually included, and a standing disclosure explains exactly what is sent.
// History lives in component state only (session-local, capped at 20).

const HISTORY_CAP = 20;

const SECTION_LABELS = {
  overview_summary: ['Overview summary', 'תמצית הסקירה'],
  schedule_freshness: ['Schedule freshness', 'טריות הלוח'],
  yield_totals: ['Yield totals', 'סיכומי תשואה'],
  recommendations: ['Recommendations', 'המלצות'],
  settings: ['Saved settings', 'הגדרות שמורות'],
  counts: ['Plan counts', 'ספירות התוכנית'],
};

const ABSENT_SUFFIX = ' (absent)';

function sourceLabel(source, locale) {
  const raw = String(source || '');
  const absent = raw.endsWith(ABSENT_SUFFIX);
  const key = absent ? raw.slice(0, -ABSENT_SUFFIX.length) : raw;
  const pair = SECTION_LABELS[key];
  const base = pair ? pageText(locale, pair[0], pair[1]) : key;
  return absent ? `${base} ${pageText(locale, '(unavailable)', '(לא זמין)')}` : base;
}

function timeLabel(iso, locale) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString(locale === 'he' ? 'he-IL' : 'en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });
}

export default function AssistantPanel({ locale, notify }) {
  const he = locale === 'he';
  const [status, setStatus] = useState(null);
  const [statusState, setStatusState] = useState('loading');
  const [question, setQuestion] = useState('');
  const [thread, setThread] = useState([]);
  const [asking, setAsking] = useState(false);
  const idRef = useRef(0);
  const threadRef = useRef(null);

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/assistant/status`)
      .then((response) => {
        if (!response.ok) throw new Error(String(response.status));
        return response.json();
      })
      .then((body) => {
        if (!active) return;
        setStatus(body);
        setStatusState('ready');
      })
      .catch(() => {
        if (!active) return;
        setStatusState('error');
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const node = threadRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [thread, asking]);

  const appendExchange = useCallback((entry) => {
    idRef.current += 1;
    const row = {
      id: `ask-${idRef.current}`,
      at: new Date().toISOString(),
      sources: [],
      answer: null,
      error: null,
      ...entry,
    };
    setThread((prev) => [...prev, row].slice(-HISTORY_CAP));
  }, []);

  const ask = useCallback(async () => {
    const trimmed = question.trim();
    if (!trimmed || asking) return;
    setAsking(true);
    try {
      const response = await fetch(`${API_BASE}/api/assistant/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed }),
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = body && body.detail ? String(body.detail) : `${response.status}`;
        appendExchange({ question: trimmed, error: detail });
      } else if (body && body.available === false) {
        setStatus((prev) => ({ ...(prev || {}), available: false, reason: body.error }));
        appendExchange({ question: trimmed, error: String(body.error || '') });
      } else if (body) {
        appendExchange({
          question: trimmed,
          answer: body.answer || null,
          error: body.error || null,
          sources: (body.grounding && body.grounding.sources) || [],
          at: (body.grounding && body.grounding.generated_at) || new Date().toISOString(),
        });
      }
      setQuestion('');
    } catch (error) {
      notify(`Assistant request failed (${error.message}).`, `הפנייה לעוזר נכשלה (${error.message}).`);
    } finally {
      setAsking(false);
    }
  }, [question, asking, appendExchange, notify]);

  function onComposerKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      ask();
    }
  }

  const unavailable = statusState === 'ready' && status && status.available === false;
  const reasonLabel = status && status.reason === 'API key not configured'
    ? pageText(locale, 'The API key is not configured on the server.', 'מפתח ה-API אינו מוגדר בשרת.')
    : String((status && status.reason) || '');

  return (
    <section className="page-panel assistant-panel">
      <div className="panel-head">
        <h2>{pageText(locale, 'AI assistant', 'עוזר AI')}</h2>
        <span>{pageText(locale, 'Grounded answers about the saved schedule and plan', 'תשובות מבוססות נתונים על הלוח והתוכנית השמורים')}</span>
      </div>

      <div className="assistant-statusline">
        <Bot size={14} />
        {statusState === 'loading' ? (
          <span>{pageText(locale, 'Checking assistant availability', 'בודק את זמינות העוזר')}</span>
        ) : statusState === 'error' ? (
          <span>{pageText(locale, 'Could not reach the Kairos server to check availability.', 'לא ניתן להגיע לשרת Kairos כדי לבדוק זמינות.')}</span>
        ) : unavailable ? (
          <span>{pageText(locale, 'Not available', 'לא זמין')}</span>
        ) : (
          <span>{pageText(locale, 'Connected', 'מחובר')}</span>
        )}
        {status && status.model ? <code dir="ltr">{status.model}</code> : null}
      </div>

      <p className="assistant-disclosure">
        {pageText(locale,
          'Answers are generated from the current saved data. When you ask a question, only the compact data summary named in the sources line is sent to Anthropic, and no other data leaves this machine.',
          'התשובות נוצרות מהנתונים השמורים הנוכחיים. בשליחת שאלה נשלח אל Anthropic רק תקציר הנתונים הקומפקטי ששמותיו מופיעים בשורת המקורות, ושום נתון אחר אינו יוצא מהמחשב הזה.')}
      </p>

      {unavailable ? (
        <div className="assistant-status-strip" role="status">
          <strong>{pageText(locale, 'The assistant is not available.', 'העוזר אינו זמין.')}</strong>
          <span>{reasonLabel}</span>
          <span>
            {pageText(locale,
              'To enable it, set the ANTHROPIC_API_KEY or KAIROS_ASSISTANT_API_KEY environment variable and restart the server.',
              'להפעלה, הגדירו את משתנה הסביבה ANTHROPIC_API_KEY או KAIROS_ASSISTANT_API_KEY והפעילו מחדש את השרת.')}
          </span>
        </div>
      ) : null}

      <div className="assistant-thread" ref={threadRef}>
        {thread.length === 0 && !asking ? (
          <div className="assistant-empty">
            {pageText(locale, 'No questions asked yet in this session.', 'עוד לא נשאלו שאלות בהפעלה הנוכחית.')}
          </div>
        ) : null}
        {thread.map((entry) => (
          <article className="assistant-exchange" key={entry.id}>
            <p className="assistant-question">{entry.question}</p>
            {entry.answer ? <p className="assistant-answer">{entry.answer}</p> : null}
            {entry.error ? <p className="assistant-error">{entry.error}</p> : null}
            <footer className="assistant-meta">
              {entry.sources.length ? (
                <span>
                  {pageText(locale, 'Based on: ', 'מבוסס על: ')}
                  {entry.sources.map((source) => sourceLabel(source, locale)).join(', ')}
                </span>
              ) : null}
              <time dir="ltr">{timeLabel(entry.at, locale)}</time>
            </footer>
          </article>
        ))}
        {asking ? (
          <div className="assistant-thinking">
            {pageText(locale, 'Computing an answer from the saved data...', 'מחשב תשובה מהנתונים השמורים...')}
          </div>
        ) : null}
      </div>

      <div className="assistant-composer">
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={onComposerKeyDown}
          rows={2}
          maxLength={2000}
          dir={he ? 'rtl' : 'ltr'}
          placeholder={pageText(locale, 'Ask about the weekly plan, in Hebrew or English', 'שאלו על התוכנית השבועית, בעברית או באנגלית')}
          disabled={asking || unavailable}
          aria-label={pageText(locale, 'Question for the assistant', 'שאלה לעוזר')}
        />
        <Button
          variant="contained"
          size="small"
          onClick={ask}
          disabled={asking || unavailable || !question.trim()}
          startIcon={<Send size={14} />}
        >
          {pageText(locale, 'Ask', 'שאלו')}
        </Button>
      </div>
      <p className="assistant-hint">
        {pageText(locale,
          'Enter sends, Shift+Enter adds a line. History is kept for this session only, up to 20 exchanges.',
          'מקש Enter שולח, Shift+Enter יורד שורה. ההיסטוריה נשמרת להפעלה הנוכחית בלבד, עד 20 שאלות.')}
      </p>
    </section>
  );
}
