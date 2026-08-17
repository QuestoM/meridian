import React from 'react';
import { Code, Name, Prose } from '../shell/bidi';
import { Status } from '../studio';
import { pageText } from '../shell/format';
import { MONEY_DESCRIBERS } from './term-language-money';
import { PROCESS_TERMS, TERM_DESCRIBERS, processDescriber } from './term-language-terms';
import { fallbackDescription, scopeLines, windowPhrase } from './term-language';
import { rankCopy, statusCopy, termName } from './trade-terms';

// One proposed term, in the language a commercial director reads.
//
// THE DIVISION OF LABOUR, which matters because two different things are being
// asserted and only one of them is this file's to assert.
//
// WHAT THE CLAUSE SAYS comes from the describers in term-language-*.js: the
// parameters turned into a sentence, the rate card rendered as a rate card, and
// each field the document did not supply named as a gap. Bilingual, because the
// reviewer may be reading either locale.
//
// WHAT IT WILL DO comes from the SERVER (`effects` on the proposal payload),
// where kairos/trade/explain.py runs the real compiler and reports its verdict:
// which live rules this term will create, or the reason it will not act at all.
// That sentence is authored in Hebrew by the engine and is rendered here through
// Prose, untranslated. This surface must never re-derive it: a UI that guesses
// an effect the compiler will not produce is the one lie this screen cannot
// afford, and the compiler is the only thing that knows.

function describe(instance, locale) {
  const termId = String(instance.term_id || '');
  const params = instance.params && typeof instance.params === 'object' ? instance.params : {};
  try {
    if (PROCESS_TERMS.includes(termId)) return processDescriber(params, locale);
    const describer = MONEY_DESCRIBERS[termId] || TERM_DESCRIBERS[termId];
    if (describer) return describer(params, locale);
  } catch {
    // A describer that throws on an unexpected shape must not take the review
    // screen down with it. The raw fields are still readable, and the fallback
    // says out loud that no sentence was written.
    return fallbackDescription(instance, locale);
  }
  return fallbackDescription(instance, locale);
}

const MECHANISM_TONE = {
  blocks: 'danger',
  warns: 'warning',
  prices: 'positive',
  steers: 'info',
  measures: 'info',
  settles: 'positive',
  records: 'neutral',
  inert: 'warning',
};

const MECHANISM_EN = {
  blocks: 'Blocks placement',
  warns: 'Warns',
  prices: 'Changes price',
  steers: 'Steers placement',
  measures: 'Measured continuously',
  settles: 'Enters settlement',
  records: 'Recorded only',
  inert: 'Will not act automatically',
};

export function MechanismChip({ effect, locale }) {
  if (!effect || !effect.mechanism) return null;
  const tone = MECHANISM_TONE[effect.mechanism] || 'neutral';
  const label = locale === 'he'
    ? effect.mechanism_he
    : MECHANISM_EN[effect.mechanism] || effect.mechanism;
  return <Status status={tone} className="trade-chip">{label}</Status>;
}

function ValueRow({ entry, locale }) {
  if (entry.missing) {
    return (
      <div className="trade-field trade-field--missing">
        <span className="trade-field__label">{entry.label}</span>
        <span className="trade-field__value">
          {pageText(locale, 'the document does not state it', 'המסמך אינו נוקב בזה')}
        </span>
      </div>
    );
  }
  return (
    <div className="trade-field">
      <span className="trade-field__label">{entry.raw ? <Code>{entry.label}</Code> : entry.label}</span>
      {entry.quote
        ? <Prose as="span" className="trade-field__value trade-field__value--quote">{entry.value}</Prose>
        : <Name className="trade-field__value">{entry.value}</Name>}
    </div>
  );
}

function TermTable({ table, locale }) {
  if (!table || !Array.isArray(table.rows) || table.rows.length === 0) return null;
  return (
    <table className="trade-term-table">
      <caption>{table.caption}</caption>
      <thead>
        <tr>
          {table.columns.map((column) => (
            <th key={column.key} scope="col" className={column.numeric ? 'is-numeric' : undefined}>
              {column.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {table.rows.map((entry, index) => (
          <tr key={index}>
            {table.columns.map((column) => (
              <td key={column.key} className={column.numeric ? 'is-numeric' : undefined}>
                {entry[column.key] === null || entry[column.key] === undefined
                  ? <span className="trade-empty">{pageText(locale, 'not stated', 'לא נקוב')}</span>
                  : <Name>{entry[column.key]}</Name>}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function TermStatement({ instance, effect, locale, showEffect = true }) {
  const description = describe(instance, locale);
  const status = statusCopy(instance.term_id, locale);
  const rank = rankCopy(instance.term_id, locale);
  const scope = scopeLines(instance.scope, locale);
  const ownWindow = windowPhrase(instance.window, locale);
  const missing = Array.isArray(instance.missing) ? instance.missing : [];
  const rows = (description.rows || []).filter(Boolean);

  return (
    <div className="trade-statement">
      <h4 className="trade-statement__lead">
        {pageText(locale, 'What the clause says', 'מה הסעיף אומר')}
      </h4>
      {description.headlineIsQuote
        ? <Prose className="trade-statement__headline trade-statement__headline--quote">{description.headline}</Prose>
        : <p className="trade-statement__headline">{description.headline}</p>}

      {rows.length > 0 ? (
        <div className="trade-fields">
          {rows.map((entry, index) => <ValueRow key={`${entry.label}-${index}`} entry={entry} locale={locale} />)}
        </div>
      ) : null}

      <TermTable table={description.table} locale={locale} />

      {scope.length > 0 ? (
        <div className="trade-statement__scope">
          <h4 className="trade-statement__lead">
            {pageText(locale, 'Who and what it applies to', 'על מי ועל מה חל')}
          </h4>
          <div className="trade-fields">
            {scope.map((entry) => (
              <div className="trade-field" key={entry.key}>
                <span className="trade-field__label">{entry.label}</span>
                <Name className="trade-field__value">{entry.value}</Name>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {ownWindow ? (
        <p className="trade-statement__window">
          {pageText(
            locale,
            `This clause has its own window, apart from the agreement's: ${ownWindow}.`,
            `לסעיף הזה חלון תוקף משלו, נפרד מזה של ההסכם: ${ownWindow}.`,
          )}
        </p>
      ) : null}

      {showEffect ? (
        <div className="trade-statement__effect">
          <h4 className="trade-statement__lead">
            {pageText(locale, 'What approving it will do', 'מה אישורו יעשה')}
          </h4>
          {effect && effect.sentence_he ? (
            <Prose className="trade-statement__effect-sentence">{effect.sentence_he}</Prose>
          ) : status ? (
            <p className="trade-statement__effect-sentence">{status.note}</p>
          ) : null}
          <div className="trade-statement__effect-meta">
            <MechanismChip effect={effect} locale={locale} />
            {status ? <Status status={status.tone} className="trade-chip">{status.label}</Status> : null}
            {rank ? <span className="trade-provenance">{rank.label}</span> : null}
          </div>
          {effect && Array.isArray(effect.will_not_act_reasons) && effect.will_not_act_reasons.length > 0 ? (
            <ul className="trade-statement__reasons">
              {effect.will_not_act_reasons.map((reason, index) => (
                <li key={index}><Prose as="span">{reason}</Prose></li>
              ))}
            </ul>
          ) : null}
          {effect && Array.isArray(effect.bound_rule_ids) && effect.bound_rule_ids.length > 0 ? (
            <p className="trade-statement__bound">
              {pageText(locale, 'Live rules it will create: ', 'כללים פעילים שייווצרו: ')}
              {effect.bound_rule_ids.map((id, index) => (
                <React.Fragment key={id}>
                  {index > 0 ? ', ' : null}
                  <Code>{id}</Code>
                </React.Fragment>
              ))}
            </p>
          ) : null}
        </div>
      ) : null}

      {missing.length > 0 ? (
        <p className="trade-statement__gaps" role="note">
          {pageText(
            locale,
            `The document did not supply ${missing.length} required field${missing.length === 1 ? '' : 's'} for ${termName(instance.term_id, locale)}: `,
            `המסמך לא מספק ${missing.length === 1 ? 'שדה חובה אחד' : `${missing.length} שדות חובה`} עבור ${termName(instance.term_id, locale)}: `,
          )}
          {missing.map((field, index) => (
            <React.Fragment key={field}>
              {index > 0 ? ', ' : null}
              <Code>{field}</Code>
            </React.Fragment>
          ))}
        </p>
      ) : null}
    </div>
  );
}

export default TermStatement;
