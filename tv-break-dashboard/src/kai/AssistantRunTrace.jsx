import React, { useEffect, useState } from 'react';
import { ChevronDown, ChevronRight, Square } from 'lucide-react';
import { pageText } from '../shell/surface-helpers';
import { Figure, Code, Name } from '../shell/bidi';
import { Pressable } from '../studio/dom-controls';

// What Mabat is doing right now, and what it did.
//
// The reference agent prints a run trace headed by the elapsed time and lists
// what it touched, so a long run reads as work rather than as a hang. Mabat's
// server streams the same facts: a stage frame the moment the request is
// accepted, one per model turn, and one step per tool with the source it read.
// Nothing here is invented. An unknown tool prints its own name rather than a
// friendly guess, and the clock is the browser's own measurement from the send.

const STAGE_LABELS = {
  accepted: ['Request received', 'הבקשה התקבלה'],
  reading: ['Opening your conversation', 'פותח את השיחה שלכם'],
  grounded: ['Reading the saved data', 'קורא את הנתונים השמורים'],
  thinking: ['Working on the answer', 'עובד על התשובה'],
  // The server caught the draft answer claiming a proposal that nothing
  // recorded, and is spending one more turn on it. Named as the check it is, so
  // the extra seconds read as work rather than as a stall.
  verifying: ['Checking that the change was really recorded', 'בודק שהשינוי אכן נרשם'],
};

const STEP_LABELS = {
  get_settings: ['Reading the saved settings', 'קורא את ההגדרות השמורות'],
  get_day_detail: ['Reading a plan day', 'קורא יום מהתוכנית'],
  list_constraints: ['Reading the placement restrictions', 'קורא את הגבלות השיבוץ'],
  list_overrides: ['Reading the manual pins', 'קורא את הנעיצות הידניות'],
  get_pricing: ['Reading the rate card', 'קורא את המחירון'],
  get_net_comparison: ['Comparing the plan to a net-focused plan', 'משווה את התוכנית לתוכנית ממוקדת נטו'],
  get_compliance: ['Checking regulatory compliance', 'בודק עמידה ברגולציה'],
  simulate_settings_change: ['Running a simulation against the optimizer', 'מריץ סימולציה מול האופטימייזר'],
  get_recommendations: ['Reading the recommendations', 'קורא את ההמלצות'],
  get_frontier: ['Reading the balance curve', 'קורא את עקומת האיזון'],
  get_audience_stability: ['Reading audience stability', 'קורא את יציבות הצפייה'],
  get_plan_days: ['Reading the plan days', 'קורא את ימי התוכנית'],
  get_schedule_freshness: ['Checking plan freshness', 'בודק את עדכניות התוכנית'],
  get_yield_per_second: ['Reading yield per second', 'קורא תשואה לשנייה'],
  get_gold_breaks: ['Reading gold breaks', 'קורא ברייקי זהב'],
  get_make_good_alerts: ['Checking make-good status', 'בודק סטטוס השלמות'],
  get_run_log_summary: ['Reading the last run summary', 'קורא את סיכום הריצה האחרונה'],
  get_upload_status: ['Checking upload status', 'בודק סטטוס העלאות'],
  get_reports_catalog: ['Reading the reports catalog', 'קורא את קטלוג הדוחות'],
  get_activity_recent: ['Reading recent activity', 'קורא פעילות אחרונה'],
  get_audience_model: ['Reading the audience model state', 'קורא את מצב מודל הקהל'],
  get_event_pipeline: ['Reading the event pipeline', 'קורא את מסלול האירועים'],
  get_agencies: ['Reading the agencies', 'קורא את הסוכנויות'],
  get_top_advertisers: ['Reading the advertiser ledger', 'קורא את ספר המפרסמים'],
  get_advertiser_airings: ['Reading advertiser airing history', 'קורא את היסטוריית שידורי המפרסם'],
  get_break_pods: ['Reading the day\'s break contents', 'קורא את תוכן הברייקים של היום'],
  get_pod: ['Reading one break\'s contents', 'קורא את תוכן הברייק'],
  get_day_breaks: ['Reading the day\'s breaks', 'קורא את הברייקים של היום'],
  get_break: ['Reading one break', 'קורא ברייק אחד'],
  get_pacing_board: ['Reading the pacing board', 'קורא את לוח הקצב'],
  get_campaign_pacing: ['Reading one campaign\'s pacing', 'קורא את הקצב של קמפיין אחד'],
  get_make_good_ledger: ['Reading the make-good ledger', 'קורא את ספר פיצויי השידור'],
  list_uploads: ['Listing your uploaded files', 'מציג את הקבצים שהעליתם'],
  get_upload: ['Reading an uploaded file', 'קורא קובץ שהועלה'],
  find_advertiser: ['Matching an advertiser', 'מאתר מפרסם'],
  propose_settings_change: ['Preparing a settings change for approval', 'מכין שינוי הגדרות לאישור'],
  propose_constraint: ['Preparing a restriction for approval', 'מכין הגבלה לאישור'],
  propose_override: ['Preparing a pin for approval', 'מכין נעיצה לאישור'],
  propose_pricing_change: ['Preparing a rate-card change for approval', 'מכין שינוי מחירון לאישור'],
  propose_recompute: ['Preparing a plan run for approval', 'מכין הרצת תוכנית לאישור'],
  propose_advertiser_change: ['Preparing an advertiser change for approval', 'מכין שינוי מפרסם לאישור'],
};

export function stepLabel(tool, locale) {
  const pair = STEP_LABELS[tool];
  return pair ? pageText(locale, pair[0], pair[1]) : '';
}

// Where each step read from, said in the reader's language. The server stamps
// every tool result with a provenance source and that string is the record: it
// goes back to the model so an answer can name its own basis, so it stays in
// English there and is read here instead of printed raw under a Hebrew heading.
// Two readings drop a word the English carries on purpose: a coefficient file
// and a training gate are model words, and section 4.2's lexicon test says a run
// surface does not show them. The operator's own name for that data is used.
const SOURCE_HE = {
  'saved settings': 'ההגדרות השמורות',
  'saved weekly plan, owned channel': 'התוכנית השבועית השמורה, הערוץ שלכם',
  'saved weekly plan, owned-scope yield': 'התוכנית השבועית השמורה, התשואה בערוץ שלכם',
  'saved weekly plan, gold segments': 'התוכנית השבועית השמורה, רצועות עם ברייקי זהב',
  'stored placement constraints': 'הגבלות השיבוץ השמורות',
  'stored manual overrides': 'הנעיצות הידניות השמורות',
  'pricing hierarchy (rate card and operator overrides)': 'היררכיית התמחור: המחירון וההתאמות של המפעיל',
  'owned-channel scenario runner, net comparison': 'מריץ התרחישים על הערוץ שלכם, השוואת נטו',
  'owned-channel scenario runner, representative day': 'מריץ התרחישים על הערוץ שלכם, יום מייצג',
  'owned-channel frontier sweep': 'סריקת עקומת האיזון על הערוץ שלכם',
  'compliance verdict over the committed plan': 'בדיקת העמידה ברגולציה על התוכנית המאושרת',
  'overview recommendations, owned channel': 'ההמלצות של מסך הפתיחה, הערוץ שלכם',
  'measured coefficients artifact, level-drift monitor': 'קובץ המדידות של יציבות הצפייה',
  'schedule freshness sidecar (input fingerprints)': 'קובץ עדכניות התוכנית, לפי טביעות האצבע של הקלט',
  'pacing make-good projection': 'תחזית ההשלמות של קצב האספקה',
  'optimizer run log': 'יומן ההרצות של האופטימייזר',
  'input upload status': 'מצב הקבצים שהועלו',
  'reports catalog': 'קטלוג הדוחות',
  'activity log (metadata only)': 'יומן הפעילות, מטא-נתונים בלבד',
  'agencies store': 'מאגר הסוכנויות',
  'agencies store (record, links, conditions)': 'מאגר הסוכנויות: הרשומה, השיוכים והתנאים',
  'calendar events store': 'מאגר אירועי היומן',
  'calendar events store and rate-card activation': 'מאגר אירועי היומן והפעלת שכבת המחירון',
  'advertiser rules store': 'מאגר כללי המפרסמים',
  'advertiser rules and scoped conditions stores': 'מאגרי כללי המפרסמים והתנאים הממוקדים',
  'daily per-spot ledger (newest daily file)': 'הספר היומי לפי ספוט, הקובץ היומי העדכני',
  'all authoritative raw daily traffic files on disk (data/daily_input/Wally_*.csv)': 'כל קובצי הטראפיק היומיים הסמכותיים הזמינים בדיסק',
  'restriction preview on the owned channel: the saved weekly plan, and an optimizer run on the days it touches': 'תצוגה מקדימה של הגבלה בערוץ שלכם: התוכנית השבועית השמורה, והרצת אופטימייזר על הימים שההגבלה נוגעת בהם',
  'the account list and the broadcast licence limits': 'רשימת החשבונות ומגבלות רישיון השידור',
  'audience model artifact (models/audience_model.json) plus the activation flag': 'קובץ מודל הקהל ומצב ההפעלה שלו',
  'event pipeline snapshot (events store, pricing layer, schedule freshness, training gate)': 'תמונת מסלול האירועים: מאגר האירועים, שכבת התמחור ועדכניות התוכנית',
  'assistant uploads (own)': 'הקבצים שהעליתם',
  'daily traffic files on disk (data/daily_input), the pods of one broadcast day': 'קובצי הטראפיק היומיים בדיסק, תוכן הברייקים של יום שידור אחד',
  'daily traffic files on disk (data/daily_input), the pod of one break': 'קובצי הטראפיק היומיים בדיסק, תוכן ברייק אחד',
  'the day plan for one channel-day, re-planned live, with the saved weekly plan beside it': 'תוכנית היום של הערוץ שלכם, מתוכננת מחדש עכשיו, לצד התוכנית השבועית השמורה',
  'the day plan for one channel-day, one break': 'תוכנית היום של הערוץ שלכם, ברייק אחד',
  'the pacing board: the campaign store and the delivery ledger, owned channel': 'לוח הקצב: מאגר הקמפיינים וספר האספקה, בערוץ שלכם',
  'the pacing board and the delivery ledger, the broadcast days behind one campaign': 'לוח הקצב וספר האספקה, ימי השידור שמאחורי קמפיין אחד',
  'the make-good decision ledger': 'ספר ההחלטות של פיצויי השידור',
  'campaign store, delivery ledger and creative-assets ledger, owned channel': 'מאגר הקמפיינים, ספר האספקה וספר חומרי הקריאייטיב, בערוץ שלכם',
  'company model-candidate artifacts and adoption decision ledger': 'קובצי המודלים המועמדים של החברה וספר החלטות ההטמעה',
  'named plan-version freezes, operator-channel summaries only': 'גרסאות תוכנית שמורות בשם, סיכומי ערוץ המפעיל בלבד',
  'unknown tool': 'כלי לא מוכר',
};

const UPLOADED_FILE_PREFIX = 'uploaded file ';

// Why the grounding line carries no owned-channel count. The server sends a code
// beside the English reason for exactly the same reason the sources are mapped.
const SCOPE_REASON = {
  no_operator_channel: ['No operator channel is set in settings, so a count would be over every channel in the file.', 'לא הוגדר ערוץ מפעיל בהגדרות, ולכן ספירה כאן הייתה חוצה את כל הערוצים בקובץ.'],
  empty_plan: ['The saved weekly plan is empty.', 'התוכנית השבועית השמורה ריקה.'],
};

export function sourceLabel(source, locale) {
  const text = String(source || '');
  if (SOURCE_HE[text]) return pageText(locale, text, SOURCE_HE[text]);
  // The file's own name is data, so it can be either script and can end in a
  // neutral. It carries its own isolate instead of being interpolated into the
  // Hebrew reading, where "report (1).csv" would otherwise flip its brackets.
  if (text.startsWith(UPLOADED_FILE_PREFIX)) {
    if (locale !== 'he') return text;
    return <>{'קובץ שהעליתם: '}<Name>{text.slice(UPLOADED_FILE_PREFIX.length)}</Name></>;
  }
  return text;
}

function seconds(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)}s` : '';
}

const PLAN_STATUS = {
  fresh: ['The plan is up to date', 'התוכנית מעודכנת'],
  stale: ['The plan is out of date', 'התוכנית אינה מעודכנת'],
  unknown: ['Plan freshness is unknown', 'עדכניות התוכנית אינה ידועה'],
};

// What Mabat is grounded on for this question, printed the moment the context is
// composed, which is a fifth of a second in. Every value is copied from that
// context by the server, so this is the scope of the answer being written and
// never an estimate of it. A fact the context does not carry prints nothing.
export function GroundedOn({ facts, locale }) {
  if (!facts || typeof facts !== 'object') return null;
  const status = PLAN_STATUS[String(facts.plan_status || '')] || null;
  const chips = [];
  if (facts.channel) chips.push(<span className="asst-run-chip" key="channel"><Name>{String(facts.channel)}</Name></span>);
  if (facts.date_from && facts.date_to) {
    chips.push(
      <span className="asst-run-chip" key="window">
        <Figure>{String(facts.date_from)}</Figure>
        {pageText(locale, ' to ', ' עד ')}
        <Figure>{String(facts.date_to)}</Figure>
      </span>,
    );
  }
  if (Number.isFinite(facts.breaks)) {
    chips.push(
      <span className="asst-run-chip" key="breaks">
        <Figure>{Number(facts.breaks).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US')}</Figure>
        {pageText(locale, ' breaks', ' ברייקים')}
      </span>,
    );
  }
  if (status) chips.push(<span className="asst-run-chip" key="status">{pageText(locale, status[0], status[1])}</span>);
  const scope = SCOPE_REASON[String(facts.scope_reason_code || '')] || null;
  if (scope || facts.scope_reason) {
    chips.push(
      <span className="asst-run-chip" key="scope">
        {scope ? pageText(locale, scope[0], scope[1]) : <Name>{String(facts.scope_reason)}</Name>}
      </span>,
    );
  }
  if (!chips.length) return null;
  return (
    <div className="asst-run-grounded">
      <span className="asst-run-grounded-label">{pageText(locale, 'Grounded on', 'מבוסס על')}</span>
      {chips}
    </div>
  );
}

// The clock ticks in the browser, from the moment the question was sent, so it
// is the same number a stopwatch beside the screen would show.
export function useElapsed(active, startedAt) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return undefined;
    setNow(Date.now());
    const id = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(id);
  }, [active, startedAt]);
  if (!active || !startedAt) return 0;
  return Math.max(0, (now - startedAt) / 1000);
}

// The budget JS-10 sets for a whole ask. Past it the run says so and points at
// the control that ends it, rather than letting the clock run without comment.
export const LONG_RUN_SECONDS = 45;

export default function AssistantRunTrace({ locale, live, elapsed, onStop }) {
  const [open, setOpen] = useState(true);
  if (!live) return null;
  const stage = live.stage && STAGE_LABELS[live.stage.stage] ? STAGE_LABELS[live.stage.stage] : null;
  const headline = stage ? pageText(locale, stage[0], stage[1]) : pageText(locale, 'Working on the answer', 'עובד על התשובה');
  const steps = Array.isArray(live.steps) ? live.steps : [];
  const deadline = Number.isFinite(live.deadlineSeconds) ? Math.round(live.deadlineSeconds) : null;
  return (
    <div className="asst-run" role="status" aria-live="polite">
      <div className="asst-run-head">
        <Pressable type="button" className="asst-run-toggle" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
          {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          <span className="asst-run-dots" aria-hidden="true"><span /><span /><span /></span>
          <span>{headline}</span>
          <time className="asst-run-clock"><Figure>{seconds(elapsed)}</Figure></time>
        </Pressable>
        {onStop ? (
          <Pressable type="button" className="asst-run-stop" onClick={onStop}>
            <Square size={11} />
            {pageText(locale, 'Stop', 'עצירה')}
          </Pressable>
        ) : null}
      </div>
      {live.facts ? <GroundedOn facts={live.facts} locale={locale} /> : null}
      {/* Why this run is taking an extra turn. It stays up for the rest of the
          run rather than flashing past, because the seconds it costs are the
          operator's and they are owed the reason for them. */}
      {live.verifying ? (
        <p className="asst-run-long">
          {pageText(locale, 'The first draft said a proposal was recorded when nothing was, so it is being written again.', 'הטיוטה הראשונה אמרה שנרשמה הצעה בזמן שלא נרשמה דבר, ולכן היא נכתבת מחדש.')}
        </p>
      ) : null}
      {elapsed > LONG_RUN_SECONDS ? (
        <p className="asst-run-long">
          {deadline
            ? pageText(locale, `This is taking longer than usual. It stops on its own after ${deadline} seconds, and Stop ends it now.`, `זה נמשך יותר מהרגיל. ההרצה נעצרת מעצמה אחרי ${deadline} שניות, וכפתור עצירה מסיים אותה עכשיו.`)
            : pageText(locale, 'This is taking longer than usual. Stop ends it now.', 'זה נמשך יותר מהרגיל. כפתור עצירה מסיים אותה עכשיו.')}
        </p>
      ) : null}
      {open && steps.length ? (
        <ol className="asst-run-steps">
          {steps.map((step, index) => (
            <li key={`${step.tool}-${index}`} className={step.ok === false ? 'fail' : ''}>
              <span>{stepLabel(step.tool, locale) || pageText(locale, 'Reading saved data', 'קורא נתונים שמורים')}</span>
              {stepLabel(step.tool, locale) ? null : <Code>{String(step.tool || '')}</Code>}
              {step.source ? <span className="asst-run-source">{sourceLabel(step.source, locale)}</span> : null}
              <time><Figure>{seconds(step.elapsed_seconds)}</Figure></time>
            </li>
          ))}
        </ol>
      ) : null}
      {open && !steps.length ? (
        <p className="asst-run-empty">
          {pageText(locale, 'No saved data has been read yet for this question.', 'עוד לא נקראו נתונים שמורים עבור השאלה הזו.')}
        </p>
      ) : null}
    </div>
  );
}
