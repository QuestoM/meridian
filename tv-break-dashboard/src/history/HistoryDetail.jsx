import React, { useEffect, useState } from 'react';
import { ChevronDown, ChevronUp, X } from 'lucide-react';
import { formatCurrency, formatNumber, formatPercent, pageText } from '../shell/format';
import HistoryRestore from './HistoryRestore';
import { fetchRun } from './history-api';
import {
  ACTION_DOORS,
  ACTION_LABELS,
  FILE_LABELS,
  KIND_LABELS,
  RUN_FIELDS,
  SIGN_IN_LABELS,
  actorHint,
  actorLabel,
  forceLabel,
  forceUnit,
  pair,
  stampLabel,
} from './history-labels';
import { genreLabel } from '../vocabulary.js';
import './history-detail.css';

// The opened entry. The header carries the position in the set and the two
// arrows that walk it, so the whole filtered timeline can be read from inside a
// record without going back, which is the mechanic Linear's issue detail uses.
//
// Every kind resolves to something a reader can act on or navigate to: a
// restore point to its diff and its restore, a run to what it produced and what
// moved since the run before it, a change to the surface it changed, a restore
// to the point that undoes it.

// A ratio in the zero-to-one band needs four decimals or a real move rounds to
// zero: the objective on the two runs measured here is 0.5404 against 0.5293,
// and one decimal renders both as 0.5 and their difference as -0.
function formatRatio(value, locale) {
  const number = Number(value);
  if (!Number.isFinite(number)) return pageText(locale, 'Not recorded', 'לא נרשם');
  return number.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US', {
    minimumFractionDigits: 4, maximumFractionDigits: 4,
  });
}

function sized(value, unit, locale) {
  if (unit === 'money') return formatCurrency(value, locale);
  if (unit === 'percent') return formatPercent(value, locale);
  if (unit === 'ratio') return formatRatio(value, locale);
  return formatNumber(value, locale);
}

function fieldValue(field, unit, locale) {
  if (field === undefined || field === null) return pageText(locale, 'Not recorded', 'לא נרשם');
  return sized(field, unit, locale);
}

// A recorded guardrail or assumption, read in its own unit. A fraction between
// zero and one is a share and reads as one; a ratio keeps its decimals because
// rounding one would hide a real difference between two runs; a list is its
// members. An unrecognised key keeps its recorded value verbatim rather than
// being coerced into a shape it may not have.
function forceValue(key, value, locale) {
  const unit = forceUnit(key);
  if (Array.isArray(value)) return value.map((item) => genreLabel(String(item), locale === 'he' ? 'he' : 'en') || String(item)).join(', ');
  // The retention floor is stored as a share of one and every other percentage
  // in this product is stored already scaled, so this is the one place that
  // scales. Without it a floor of 0.72 reads as 0.7 percent.
  if (unit === 'fraction') return formatPercent(Number(value) * 100, locale);
  if (unit === 'ratio') return formatRatio(value, locale);
  if (unit === 'count') return formatNumber(value, locale);
  return String(value);
}

function DeltaCell({ field, unit, locale }) {
  if (!field || field.state !== 'measured') {
    return <span className="hist-delta none">{pageText(locale, 'Not recorded on both runs', 'לא נרשם בשתי ההרצות')}</span>;
  }
  const value = Number(field.delta);
  if (!Number.isFinite(value) || value === 0) {
    return <span className="hist-delta flat">{pageText(locale, 'No change', 'ללא שינוי')}</span>;
  }
  const sign = value > 0 ? '+' : '-';
  return <span className={`hist-delta ${value > 0 ? 'up' : 'down'}`} dir="ltr">{`${sign}${sized(Math.abs(value), unit, locale)}`}</span>;
}

function RunDetail({ entry, locale }) {
  const runId = String((entry.facts || {}).run_id || '');
  const [state, setState] = useState({ status: 'loading', data: null, error: '' });

  useEffect(() => {
    let active = true;
    setState({ status: 'loading', data: null, error: '' });
    fetchRun(runId).then((result) => {
      if (!active) return;
      if (result.ok) setState({ status: 'ready', data: result.data, error: '' });
      else setState({ status: 'error', data: null, error: result.error });
    });
    return () => { active = false; };
  }, [runId]);

  if (state.status === 'loading') return <p className="hist-empty">{pageText(locale, 'Reading the run record', 'קורא את רשומת ההרצה')}</p>;
  if (state.status === 'error') return <p className="hist-empty warn" dir="auto">{pageText(locale, `The run record could not be read. ${state.error}`, `לא ניתן לקרוא את רשומת ההרצה. ${state.error}`)}</p>;

  const run = state.data;
  const summary = run.summary || {};
  const comparison = run.comparison || {};
  const fields = comparison.fields || {};
  const scope = [run.channel, run.day].filter(Boolean).join(' / ');

  return (
    <div className="hist-run">
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'Scope', 'היקף')}</span>
        <span dir="auto">{scope || pageText(locale, 'The whole saved plan', 'כל התוכנית השמורה')}</span>
      </div>
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'Engine version', 'גרסת מנוע')}</span>
        <span dir="ltr">{run.engine_version}</span>
      </div>
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'Run id', 'מזהה הרצה')}</span>
        <code dir="ltr">{run.run_id}</code>
      </div>

      <h4 className="hist-detail-h">{pageText(locale, 'What this run produced', 'מה ההרצה הזו הפיקה')}</h4>
      <p className="hist-basis" dir="auto">{pageText(locale, `Every figure below is this run's own recorded outcome over ${scope}.`, `כל מספר כאן הוא התוצאה שההרצה עצמה רשמה, על ${scope}.`)}</p>
      <div className="hist-run-grid">
        <div className="hist-run-head">
          <span>{pageText(locale, 'Figure', 'מדד')}</span>
          <span>{pageText(locale, 'This run', 'ההרצה הזו')}</span>
          <span>{pageText(locale, 'Change from the run before', 'שינוי מההרצה הקודמת')}</span>
        </div>
        {RUN_FIELDS.map(([key, en, he, unit]) => (
          <div className="hist-run-row" key={key}>
            <span dir="auto">{pageText(locale, en, he)}</span>
            <span dir="ltr">{fieldValue(key === 'segment_count' ? run.segment_count : summary[key], unit, locale)}</span>
            {comparison.state === 'measured'
              ? <DeltaCell field={fields[key]} unit={unit} locale={locale} />
              : <span className="hist-delta none">{pageText(locale, 'No earlier run on this scope', 'אין הרצה קודמת על ההיקף הזה')}</span>}
          </div>
        ))}
      </div>

      <h4 className="hist-detail-h">{pageText(locale, 'What it read', 'מה היא קראה')}</h4>
      <div className="hist-kv">
        {Object.entries(run.inputs || {}).map(([name, checksum]) => (
          <div className="hist-detail-line" key={name}>
            <span className="hist-detail-key" dir="ltr">{name}</span>
            <code dir="ltr">{checksum || pageText(locale, 'The file was not there', 'הקובץ לא היה שם')}</code>
          </div>
        ))}
      </div>

      <h4 className="hist-detail-h">{pageText(locale, 'What was in force', 'מה היה בתוקף')}</h4>
      <div className="hist-kv">
        {Object.entries({ ...(run.guardrails || {}), ...(run.assumptions || {}) }).map(([name, value]) => (
          <div className="hist-detail-line" key={name}>
            <span className="hist-detail-key" dir="auto">{forceLabel(name, locale)}</span>
            <span dir="auto">{forceValue(name, value, locale)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ChangeDetail({ entry, locale }) {
  const facts = entry.facts || {};
  const door = ACTION_DOORS[facts.action];
  const status = Number(facts.status);
  const members = Array.isArray(entry.members) ? entry.members : [entry];
  const preview = entry.kind === 'preview';
  return (
    <div className="hist-kv">
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'What', 'מה')}</span>
        <span dir="auto">{pair(ACTION_LABELS, facts.action, locale) || pair(ACTION_LABELS, 'other', locale)}</span>
      </div>
      {members.length > 1 ? (
        <div className="hist-detail-line">
          <span className="hist-detail-key">{pageText(locale, 'How many', 'כמה')}</span>
          <span dir="auto">{pageText(locale, `${members.length} of these in a row, from ${stampLabel(entry.oldestTs, locale)} to ${stampLabel(entry.ts, locale)}.`, `${members.length} ברצף, מ-${stampLabel(entry.oldestTs, locale)} עד ${stampLabel(entry.ts, locale)}.`)}</span>
        </div>
      ) : null}
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'Request', 'בקשה')}</span>
        <code dir="ltr">{`${facts.method || ''} ${facts.path || ''}`.trim()}</code>
      </div>
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'Result', 'תוצאה')}</span>
        <span dir="ltr">{Number.isFinite(status) ? status : pageText(locale, 'Not recorded', 'לא נרשם')}</span>
      </div>
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'Took', 'ארך')}</span>
        <span dir="ltr">{facts.duration_ms === null || facts.duration_ms === undefined ? pageText(locale, 'Not recorded', 'לא נרשם') : `${formatNumber(facts.duration_ms, locale)} ms`}</span>
      </div>
      {preview ? (
        <p className="hist-note" dir="auto">{pageText(locale, 'Nothing was saved. This act computed an answer from the saved state and left it exactly as it was, so there is nothing here to put back.', 'לא נשמר דבר. הפעולה חישבה תשובה מהמצב השמור והשאירה אותו בדיוק כפי שהיה, ולכן אין כאן מה להחזיר.')}</p>
      ) : null}
      <p className="hist-note" dir="auto">{pageText(locale, 'The recorder stores metadata only: no request body, no query string and no credential has ever entered this record.', 'הרישום שומר נתוני-על בלבד: גוף הבקשה, מחרוזת השאילתה ופרטי הזדהות מעולם לא נכנסו לרשומה הזו.')}</p>
      {members.length > 1 ? (
        <div className="hist-members">
          <span className="hist-detail-key">{pageText(locale, 'Every one of them', 'כל אחת מהן')}</span>
          {members.map((member) => (
            <div className="hist-member" key={member.id}>
              <time dir="ltr">{stampLabel(member.ts, locale)}</time>
              <code dir="ltr">{`${(member.facts || {}).method || ''} ${(member.facts || {}).path || ''}`.trim()}</code>
              <span dir="ltr">{(member.facts || {}).status}</span>
            </div>
          ))}
        </div>
      ) : null}
      {door ? (
        <a className="hist-door" href={`#${encodeURIComponent(door)}`}>
          {preview
            ? pageText(locale, 'Open the surface this was computed for', 'פתחו את המסך שעבורו זה חושב')
            : pageText(locale, 'Open the surface this changed', 'פתחו את המסך שהשתנה')}
        </a>
      ) : null}
    </div>
  );
}

function RestoreDetail({ entry, locale, onOpenVersion }) {
  const facts = entry.facts || {};
  const files = (facts.restored || []).map((file) => pair(FILE_LABELS, file, locale) || file);
  return (
    <div className="hist-kv">
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'Put back', 'הוחזרו')}</span>
        <span dir="auto">{files.join(', ')}</span>
      </div>
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'From the point', 'מהנקודה')}</span>
        {facts.version_id ? (
          <button type="button" className="hist-link" onClick={() => onOpenVersion(facts.version_id)}>
            <code dir="ltr">{facts.version_id}</code>
          </button>
        ) : (
          <span dir="auto">{pageText(locale, 'Not recorded', 'לא נרשם')}</span>
        )}
      </div>
      {facts.safety_version_id ? (
        <div className="hist-detail-line">
          <span className="hist-detail-key">{pageText(locale, 'Undo it with', 'לביטול')}</span>
          <button type="button" className="hist-link" onClick={() => onOpenVersion(facts.safety_version_id)}>
            <code dir="ltr">{facts.safety_version_id}</code>
          </button>
        </div>
      ) : null}
      <p className="hist-note" dir="auto">{pageText(locale, 'The state before this restore was saved as its own point, so the restore is itself reversible.', 'המצב שלפני השחזור נשמר כנקודה נפרדת, ולכן גם השחזור עצמו הפיך.')}</p>
    </div>
  );
}

function SignInDetail({ entry, locale }) {
  const facts = entry.facts || {};
  return (
    <div className="hist-kv">
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'What', 'מה')}</span>
        <span dir="auto">{pair(SIGN_IN_LABELS, facts.event, locale)}</span>
      </div>
      <div className="hist-detail-line">
        <span className="hist-detail-key">{pageText(locale, 'Role', 'תפקיד')}</span>
        <span dir="auto">{facts.role || pageText(locale, 'Not recorded', 'לא נרשם')}</span>
      </div>
    </div>
  );
}

export default function HistoryDetail(props) {
  const { entry, locale, position, total, onStep, onClose } = props;
  if (!entry) {
    return (
      <aside className="hist-detail empty">
        <p className="hist-empty" dir="auto">{pageText(locale, 'Choose a row to see what it was, who did it and how to put it back.', 'בחרו שורה כדי לראות מה קרה, מי עשה זאת ואיך להחזיר.')}</p>
      </aside>
    );
  }
  return (
    <aside className="hist-detail" aria-label={pageText(locale, 'Entry detail', 'פרטי הרשומה')}>
      <header className="hist-detail-head">
        <div>
          <span className="hist-detail-kind">{pair(KIND_LABELS, entry.kind, locale)}</span>
          <span className="hist-detail-actor" dir="auto" title={actorHint(entry.actor, locale)}>{actorLabel(entry.actor, locale)}</span>
          <time className="hist-detail-time" dir="ltr">{stampLabel(entry.ts, locale)}</time>
        </div>
        <div className="hist-detail-walk">
          <span className="hist-counter" dir="ltr">{`${position} / ${total}`}</span>
          <button type="button" className="hist-icon-btn" onClick={() => onStep(-1)} aria-label={pageText(locale, 'Previous entry', 'הרשומה הקודמת')}><ChevronUp size={14} /></button>
          <button type="button" className="hist-icon-btn" onClick={() => onStep(1)} aria-label={pageText(locale, 'Next entry', 'הרשומה הבאה')}><ChevronDown size={14} /></button>
          <button type="button" className="hist-icon-btn" onClick={onClose} aria-label={pageText(locale, 'Close', 'סגירה')}><X size={14} /></button>
        </div>
      </header>
      {actorHint(entry.actor, locale) ? (
        <p className="hist-note hist-actor-note" dir="auto">{actorHint(entry.actor, locale)}</p>
      ) : null}
      <div className="hist-detail-body">
        {entry.kind === 'restore_point' ? <HistoryRestore {...props} /> : null}
        {entry.kind === 'run' ? <RunDetail entry={entry} locale={locale} /> : null}
        {entry.kind === 'change' || entry.kind === 'preview' ? <ChangeDetail entry={entry} locale={locale} /> : null}
        {entry.kind === 'restore' ? <RestoreDetail entry={entry} locale={locale} onOpenVersion={props.onOpenVersion} /> : null}
        {entry.kind === 'sign_in' ? <SignInDetail entry={entry} locale={locale} /> : null}
      </div>
    </aside>
  );
}
