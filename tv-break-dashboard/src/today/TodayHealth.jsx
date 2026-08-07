import React from 'react';
import { Button } from '@mui/material';
import { AlertTriangle, CheckCircle2, HelpCircle, Info } from 'lucide-react';
import { formatNumber, pageText } from '../shell/format';
import { formatDay } from '../shell/dates';
import { Name } from '../shell/bidi';
import { word } from '../vocabulary.js';

// Answer two: is anything broken.
//
// Every row is one measured field from the payload, with the figure it was
// decided on printed on the row. A row that reports a state opens what caused
// it, which is the no-dead-end rule applied to a status.
//
// The model side is one row and one sentence. A newer model version exists, and
// the date it was trained, is everything an operator can act on; the gate that
// decided it belongs to the other side of the line and never renders here.

const ICONS = { ok: CheckCircle2, attention: AlertTriangle, notice: Info, unknown: HelpCircle };

// What the row does when it is pressed, printed on the row. A status that opens
// something should say what, before it is clicked rather than after.
const OPENS = {
  operator_channel_unset: ['Open settings', 'פתחו את ההגדרות'],
  plan_out_of_date: ['Run the plan', 'הריצו את התוכנית'],
  newer_model_version: ['Run the plan', 'הריצו את התוכנית'],
  plan_current: ['Open the plan', 'פתחו את התוכנית'],
  plan_currency_unknown: ['Open the plan', 'פתחו את התוכנית'],
  licence: ['Open the checks', 'פתחו את הבדיקות'],
  inputs: ['Open the sources', 'פתחו את המקורות'],
};

function joinLabels(labels, locale) {
  const list = labels.filter(Boolean);
  if (list.length <= 1) return list.join('');
  const head = list.slice(0, -1).join(', ');
  return locale === 'he' ? `${head} ו${list[list.length - 1]}` : `${head} and ${list[list.length - 1]}`;
}

function stamp(value) {
  const text = String(value || '').slice(0, 10);
  return text ? formatDay(text) : '';
}

function rowCopy(check, locale, sourceFiles) {
  if (check.id === 'operator_channel_unset') {
    return {
      title: pageText(locale, 'The operator channel is not set', 'ערוץ המפעיל אינו מוגדר'),
      detail: pageText(locale, check.reason_en || '', check.reason_he || ''),
    };
  }
  if (check.id === 'plan_out_of_date') {
    const changed = joinLabels((check.changed || []).map((item) => pageText(locale, item.label_en, item.label_he)), locale);
    return {
      title: pageText(locale, 'Your changes are not in the plan yet', 'השינויים שלכם עדיין לא נכנסו לתוכנית'),
      detail: pageText(
        locale,
        `${changed} changed after the plan was last run on ${stamp(check.plan_run_at)}`,
        `${changed} השתנו אחרי שהתוכנית הורצה לאחרונה ב${stamp(check.plan_run_at)}`,
      ),
    };
  }
  if (check.id === 'newer_model_version') {
    return {
      title: word('state.newer_model_version', locale),
      detail: pageText(
        locale,
        `Trained on ${stamp(check.model_trained_at)}, after this plan was run on ${stamp(check.plan_run_at)}`,
        `אומנה ב${stamp(check.model_trained_at)}, אחרי שהתוכנית הזו הורצה ב${stamp(check.plan_run_at)}`,
      ),
    };
  }
  if (check.id === 'plan_current') {
    return {
      title: pageText(locale, 'The plan is current with its inputs', 'התוכנית מעודכנת מול הקלטים שלה'),
      detail: pageText(locale, `Run on ${stamp(check.plan_run_at)}`, `הורצה ב${stamp(check.plan_run_at)}`),
    };
  }
  if (check.id === 'plan_currency_unknown') {
    return {
      title: pageText(locale, 'Whether the plan is current cannot be verified', 'לא ניתן לאמת אם התוכנית מעודכנת'),
      detail: pageText(locale, 'The saved plan carries no record of the inputs it was built from', 'לתוכנית השמורה אין תיעוד של הקלטים שממנה נבנתה'),
    };
  }
  if (check.id === 'licence') {
    const breached = joinLabels(pageText(locale, check.breached_labels_en || [], check.breached_labels_he || []), locale);
    if (!check.checks_total) {
      return {
        title: pageText(locale, 'The licence has not been checked', 'הרישיון לא נבדק'),
        detail: pageText(locale, 'No compliance rule was evaluated against this plan', 'לא נבדק אף כלל רגולציה מול התוכנית הזו'),
      };
    }
    return {
      title: check.checks_breached
        ? pageText(locale, 'The plan is outside the licence', 'התוכנית חורגת מהרישיון')
        : pageText(locale, 'The plan is inside the licence', 'התוכנית בתוך הרישיון'),
      detail: check.checks_breached
        ? pageText(locale, `${check.checks_breached} of ${check.checks_total} checks breached: ${breached}`, `${check.checks_breached} מתוך ${check.checks_total} בדיקות חורגות: ${breached}`)
        : pageText(locale, `${check.checks_total} of ${check.checks_total} checks pass, ${check.profile}`, `${check.checks_total} מתוך ${check.checks_total} בדיקות עוברות, ${check.profile}`),
    };
  }
  if (check.id === 'inputs') {
    if (check.programmes === null) {
      return {
        title: pageText(locale, 'The inputs have not been counted yet', 'הקלטים עדיין לא נספרו'),
        detail: pageText(locale, 'This reading did not arrive, so nothing here is a count', 'הקריאה הזו לא הגיעה, ולכן שום דבר כאן אינו ספירה'),
      };
    }
    const filesPart = sourceFiles
      ? pageText(locale, `, ${sourceFiles.present} of ${sourceFiles.total} source files present`, `, ${sourceFiles.present} מתוך ${sourceFiles.total} קבצי מקור קיימים`)
      : '';
    return {
      title: check.missing && check.missing.length
        ? pageText(locale, 'The plan is missing an input it needs', 'לתוכנית חסר קלט שהיא זקוקה לו')
        : pageText(locale, 'The plan read every input it needs', 'התוכנית קראה את כל הקלטים שהיא זקוקה להם'),
      detail: pageText(
        locale,
        `${formatNumber(check.programmes, 'en')} programme rows, ${formatNumber(check.spots, 'en')} spot rows, ${formatNumber(check.planned_break_rows, 'en')} planned break rows${filesPart}`,
        `${formatNumber(check.programmes, 'he')} שורות תוכניות, ${formatNumber(check.spots, 'he')} שורות תשדירים, ${formatNumber(check.planned_break_rows, 'he')} שורות ברייקים מתוכננים${filesPart}`,
      ),
    };
  }
  return { title: String(check.id || ''), detail: '' };
}

export function TodayHealth({ today, locale, sourceFiles = null, onOpen }) {
  const health = today.health || {};
  const checks = Array.isArray(health.checks) ? health.checks : [];
  const attention = Number(health.attention_count || 0);
  const headline = attention === 0
    ? pageText(locale, 'Nothing needs attention', 'אין דבר שדורש טיפול')
    : attention === 1
      ? pageText(locale, 'One thing needs attention', 'נושא אחד דורש טיפול')
      : pageText(locale, `${attention} things need attention`, `${attention} נושאים דורשים טיפול`);

  return (
    <section className="page-panel today-answer today-answer-health" aria-label={pageText(locale, 'Is anything broken', 'האם משהו לא תקין')}>
      <div className="today-answer-head">
        <h2>{pageText(locale, 'Is anything broken', 'האם משהו לא תקין')}</h2>
        <span className={`today-verdict ${attention ? 'behind' : 'on_plan'}`}>{headline}</span>
      </div>
      <div className="today-health-list">
        {checks.map((check) => {
          const Icon = ICONS[check.status] || Info;
          const copy = rowCopy(check, locale, sourceFiles);
          const opens = OPENS[check.id];
          return (
            <Button className={`today-health-row ${check.status}`} type="button" key={check.id} onClick={() => onOpen && onOpen(check)}>
              <span className={`today-health-icon ${check.status}`}><Icon size={16} strokeWidth={2} /></span>
              <span className="today-health-copy">
                <strong>{copy.title}</strong>
                <Name>{copy.detail}</Name>
              </span>
              <span className="today-health-opens">{opens ? pageText(locale, opens[0], opens[1]) : ''}</span>
            </Button>
          );
        })}
      </div>
    </section>
  );
}

export default TodayHealth;
