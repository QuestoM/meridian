import React from 'react';
import { formatCurrency, formatNumber, formatPercent, pageText } from '../shell/format';
import { Figure, Name } from '../shell/bidi';
import { foldSize } from './history-fold';
import { APPLIED, actLabel, outcomeOf, outcomeWord } from './history-refused';
import {
  FILE_LABELS,
  KIND_LABELS,
  SIGN_IN_LABELS,
  SOURCE_LABELS,
  VIA_LABELS,
  actorLabel,
  clockLabel,
  pair,
  pathStem,
} from './history-labels';

// One timeline row. Nine facts on one line at one row height, which is the
// density Linear reaches and the reason its activity feed reads at a glance:
// the clock, the kind, the actor, what happened, what it happened to, the one
// number the kind carries, how it was done, whether it can be put back, and the
// single affordance hover reveals. Nothing is invented here: a fact the entry
// does not carry is simply absent from the row.
//
// What happened is told by the outcome as well as by the act. Before this round
// a refused write printed the sentence of one that succeeded and differed only
// in a small red number, so four rows a minute apart read alike while two of
// them had changed nothing at all.

function factsFor(entry, locale) {
  const facts = entry.facts || {};
  if (entry.kind === 'change' || entry.kind === 'preview') {
    const status = Number(facts.status);
    return {
      title: actLabel(facts.action, outcomeOf(entry), locale),
      target: facts.path ? pathStem(facts.path) : '',
      figure: Number.isFinite(status) && status > 0 ? String(status) : '',
      figureTone: Number.isFinite(status) && status >= 400 ? 'warn' : 'quiet',
    };
  }
  if (entry.kind === 'run') {
    // The scope is printed beside the figure, never in a tooltip: a one-day run
    // and a whole-plan run sit on the same list and their money is not the same
    // quantity, so the row that carries the figure carries what it is over.
    const span = facts.day || pageText(locale, 'the whole saved plan', 'כל התוכנית השמורה');
    const scope = [facts.channel, span].filter(Boolean).join(' / ');
    return {
      title: pageText(locale, 'Weekly plan run', 'הרצת הלוח השבועי'),
      target: scope,
      figure: facts.projected_revenue === undefined ? '' : formatCurrency(facts.projected_revenue, locale),
      figureTone: 'money',
    };
  }
  if (entry.kind === 'restore_point') {
    const source = pair(SOURCE_LABELS, facts.source, locale);
    const files = (facts.files || []).map((file) => pair(FILE_LABELS, file, locale) || file);
    return {
      title: facts.label ? String(facts.label) : source || pageText(locale, 'Restore point', 'נקודת שחזור'),
      target: files.join(', '),
      figure: '',
      figureTone: 'quiet',
    };
  }
  if (entry.kind === 'restore') {
    const files = (facts.restored || []).map((file) => pair(FILE_LABELS, file, locale) || file);
    return {
      title: pageText(locale, 'Put back', 'הוחזר'),
      target: files.join(', '),
      figure: '',
      figureTone: 'quiet',
    };
  }
  return {
    title: pair(SIGN_IN_LABELS, facts.event, locale) || pageText(locale, 'Account', 'חשבון'),
    target: facts.role ? String(facts.role) : '',
    figure: '',
    figureTone: 'quiet',
  };
}

export default function HistoryRow({ entry, locale, selected, onSelect, index }) {
  const facts = entry.facts || {};
  const view = factsFor(entry, locale);
  const via = pair(VIA_LABELS, entry.via, locale);
  const folded = foldSize(entry);
  // Only a recorded request has an outcome. A restore point and a run are read
  // from acts that already happened, so asking would answer a question the
  // entry was never about.
  const outcome = entry.kind === 'change' || entry.kind === 'preview' ? outcomeOf(entry) : '';
  const outcomeChip = outcome && outcome !== APPLIED ? outcomeWord(outcome, locale) : '';
  const blocked = entry.kind === 'restore_point' && facts.restorable === false;
  const runFigure = entry.kind === 'run' && facts.total_breaks !== undefined
    ? `${formatNumber(facts.total_breaks, locale)} ${pageText(locale, 'breaks', 'ברייקים')}`
    : '';
  const retention = entry.kind === 'run' && facts.average_retention !== undefined
    ? formatPercent(facts.average_retention, locale)
    : '';

  return (
    <div
      className={`hist-row${selected ? ' selected' : ''}`}
      role="option"
      aria-selected={selected}
      tabIndex={selected ? 0 : -1}
      data-index={index}
      data-kind={entry.kind}
      data-outcome={outcome || undefined}
      onClick={() => onSelect(entry)}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onSelect(entry);
        }
      }}
    >
      <time className="hist-row-time"><Figure>{clockLabel(entry.ts, locale)}</Figure></time>
      <span className={`hist-dot k-${entry.kind}`} aria-hidden="true" />
      <span className="hist-row-kind">{pair(KIND_LABELS, entry.kind, locale)}</span>
      <span className="hist-row-actor"><Name>{actorLabel(entry.actor, locale)}</Name></span>
      <span className="hist-row-title"><Name>{view.title}</Name></span>
      {folded > 1 ? <span className="hist-fold"><Figure>{`x${formatNumber(folded, locale)}`}</Figure></span> : null}
      {view.target ? <span className="hist-row-target"><Name>{view.target}</Name></span> : null}
      <span className="hist-row-tail">
        {outcomeChip ? <span className="hist-chip refused">{outcomeChip}</span> : null}
        {retention ? <span className="hist-chip quiet"><Figure>{retention}</Figure></span> : null}
        {runFigure ? <span className="hist-chip quiet">{runFigure}</span> : null}
        {view.figure ? <span className={`hist-figure ${view.figureTone}`}><Figure>{view.figure}</Figure></span> : null}
        {via ? <span className="hist-chip via">{via}</span> : null}
        {blocked ? <span className="hist-chip warn">{pageText(locale, 'Not restorable', 'לא ניתן לשחזור')}</span> : null}
        <span className="hist-row-open">{pageText(locale, 'Open', 'פתיחה')}</span>
      </span>
    </div>
  );
}
