import React, { useState } from 'react';
import { Card, CardBody, Status } from '../studio';
import { Button } from '../studio/actions';
import {
  Ban, Check, CircleSlash2, Pencil, Scale, TriangleAlert,
} from 'lucide-react';
import { Code, Figure, Name, Prose } from '../shell/bidi';
import { pageText } from '../shell/format';
import { rankCopy, statusCopy, termName } from './trade-terms';
import { scopeLines, windowPhrase } from './term-language';
import {
  confidenceLabel, confidenceTone, mechanismLabel, mechanismNote, mechanismTone,
  reviewStateLabel, reviewStateTone,
} from './trade-vocabulary';

// One proposed term, as the person signing the agreement has to read it.
//
// THE ORDER ON THIS CARD IS AN ARGUMENT. What the clause says comes first, then
// what approving it will DO, then the evidence, and only then the extracted
// fields. A reviewer shown the fields first is proofreading a data structure; a
// reviewer shown the effect first is reviewing the agreement.
//
// THE EFFECT SENTENCE IS THE BACKEND'S. `sentence_he`, `scope_he` and
// `mechanism_he` come from the compiler's own verdict on the CURRENT reviewed
// state, so an edit changes the sentence. Re-deriving any of it here would give
// the screen a second opinion about what the engine is going to do, and the
// screen's opinion would be the wrong one.
//
// A TERM THAT WILL NOT ACT SAYS SO LOUDEST. `will_not_act_reasons` is the single
// most consequential thing on this surface: a clause that looks binding, reads as
// binding, and moves nothing is how a channel discovers in month four that it
// gave something away for free. It renders as a danger block with every reason,
// never as a quiet footnote.

function ParamValue({ value }) {
  if (value === null || value === undefined || value === '') {
    return <span className="trd-param-missing">{'—'}</span>;
  }
  if (typeof value === 'boolean') {
    return <Code>{String(value)}</Code>;
  }
  if (typeof value === 'number') {
    return <Figure>{String(value)}</Figure>;
  }
  if (Array.isArray(value) || typeof value === 'object') {
    return <Code className="trd-param-json">{JSON.stringify(value, null, 1)}</Code>;
  }
  return <Name>{String(value)}</Name>;
}

// The extracted fields, exactly as they were extracted. This is the audit view
// and it is labelled as one: the parameter names are the schema's own, in
// English, and nothing here paraphrases a value the document wrote.
function ParamTable({ params, locale }) {
  const entries = Object.entries(params || {});
  if (entries.length === 0) {
    return (
      <p className="trd-field-hint">
        {pageText(locale, 'This term carries no parameters.', 'למונח הזה אין פרמטרים.')}
      </p>
    );
  }
  return (
    <dl className="trd-params">
      {entries.map(([key, value]) => (
        <React.Fragment key={key}>
          <dt><Code>{key}</Code></dt>
          <dd><ParamValue value={value} /></dd>
        </React.Fragment>
      ))}
    </dl>
  );
}

export default function ReviewTermCard({
  term, locale, canEdit, highlighted, busy, onJumpToClause, onConfirm, onEdit, onReject,
  onResolveConflict,
}) {
  const [showFields, setShowFields] = useState(false);
  const effect = term.effect;
  const taxonomy = statusCopy(term.term_id, locale);
  const rank = rankCopy(term.term_id, locale);
  const scope = scopeLines(term.scope, locale);
  const termWindow = windowPhrase(term.window, locale);
  const inert = Boolean(effect && effect.mechanism === 'inert');
  const params = term.editedParams || term.params;
  const undecided = term.state === 'proposed';

  return (
    <Card
      className="trd-term"
      data-highlighted={highlighted ? 'true' : undefined}
      data-mechanism={effect ? effect.mechanism : 'none'}
      id={`trd-term-${term.instance_id}`}
    >
      <CardBody>
        <div className="trd-term-head">
          <div className="trd-term-title">
            <h4><Name>{termName(term.term_id, locale)}</Name></h4>
            <Code className="trd-term-id">{term.instance_id}</Code>
          </div>
          <div className="trd-term-marks">
            <Status status={reviewStateTone(term.state)}>{reviewStateLabel(term.state, locale)}</Status>
            {effect ? (
              <Status
                status={mechanismTone(effect.mechanism)}
                icon={inert ? <CircleSlash2 size={13} aria-hidden="true" /> : null}
              >
                {mechanismLabel(effect.mechanism, locale, effect.mechanism_he)}
              </Status>
            ) : null}
            {term.confidence ? (
              <Status status={confidenceTone(term.confidence)}>
                {confidenceLabel(term.confidence, locale)}
              </Status>
            ) : null}
          </div>
        </div>

        {effect && effect.sentence_he ? (
          <Prose className="trd-term-sentence">{effect.sentence_he}</Prose>
        ) : (
          <p className="trd-field-hint">
            {term.state === 'rejected'
              ? pageText(
                locale,
                'This term was rejected, so the engine computes no effect for it. Nothing will act on it.',
                'המונח נדחה, ולכן המנוע אינו מחשב לו השפעה. דבר לא יפעל לפיו.',
              )
              : pageText(
                locale,
                'The engine returned no effect sentence for this term.',
                'המנוע לא החזיר משפט השפעה למונח הזה.',
              )}
          </p>
        )}

        {effect && effect.scope_he ? (
          <Prose className="trd-term-scope">{effect.scope_he}</Prose>
        ) : null}

        {inert ? (
          <div className="trd-inert" role="note">
            <p className="trd-inert-lead">
              <TriangleAlert size={15} aria-hidden="true" />
              {pageText(
                locale,
                'Approving this changes nothing on its own. Here is exactly why:',
                'אישור הסעיף הזה אינו משנה דבר מעצמו. וזאת הסיבה המדויקת:',
              )}
            </p>
            <ul>
              {(effect.will_not_act_reasons || []).map((reason) => (
                <li key={reason}><Prose as="span">{reason}</Prose></li>
              ))}
              {(effect.will_not_act_reasons || []).length === 0 ? (
                <li>
                  <span>
                    {pageText(
                      locale,
                      'The engine reports no reason. That silence is itself the finding.',
                      'המנוע אינו מדווח סיבה. השתיקה הזו היא עצמה הממצא.',
                    )}
                  </span>
                </li>
              ) : null}
            </ul>
          </div>
        ) : null}

        {effect && !inert && effect.bound_rule_ids && effect.bound_rule_ids.length > 0 ? (
          <p className="trd-term-rules">
            <span className="trd-card-label">{pageText(locale, 'Rules it will write', 'כללים שייכתבו')}</span>
            {effect.bound_rule_ids.map((ruleId) => <Code key={ruleId} className="trd-id-chip">{ruleId}</Code>)}
          </p>
        ) : null}

        {effect && effect.settlement_kinds && effect.settlement_kinds.length > 0 ? (
          <p className="trd-term-rules">
            <span className="trd-card-label">{pageText(locale, 'Settlement it enters', 'התחשבנות שאליה ייכנס')}</span>
            {effect.settlement_kinds.map((kind) => <Code key={kind} className="trd-id-chip">{kind}</Code>)}
          </p>
        ) : null}

        {term.missing.length > 0 || (effect && effect.incomplete) ? (
          <div className="trd-missing" role="note">
            <TriangleAlert size={14} aria-hidden="true" />
            <span>
              {term.missing.length > 0
                ? pageText(
                  locale,
                  'The document did not supply these required fields, so they are gaps and not zeroes:',
                  'המסמך לא ספק את השדות הנדרשים האלה, ולכן הם חסרים ולא אפסים:',
                )
                : pageText(
                  locale,
                  'The engine reports this term as incomplete.',
                  'המנוע מדווח שהמונח חסר פרטים.',
                )}
            </span>
            {term.missing.map((field) => <Code key={field} className="trd-id-chip">{field}</Code>)}
          </div>
        ) : null}

        {term.conflict ? (
          <div className={term.conflict.open ? 'trd-conflict open' : 'trd-conflict'} role="note">
            <p className="trd-conflict-lead">
              <Scale size={15} aria-hidden="true" />
              {term.conflict.open
                ? pageText(locale, 'This term contradicts another clause, and nobody has decided which one governs.', 'המונח הזה סותר סעיף אחר, ואף אחד לא הכריע מי מהם קובע.')
                : pageText(locale, 'This term contradicted another clause. The contradiction is settled.', 'המונח הזה סתר סעיף אחר. הסתירה הוכרעה.')}
            </p>
            {term.conflict.contested ? <Prose className="trd-conflict-body">{term.conflict.contested}</Prose> : null}
            {term.conflict.explanationHe ? (
              <Prose className="trd-conflict-body">{term.conflict.explanationHe}</Prose>
            ) : null}
            {term.conflict.winner ? (
              <p className="trd-conflict-winner">
                <span className="trd-card-label">{pageText(locale, 'Governing term', 'המונח הקובע')}</span>
                <Code>{term.conflict.winner}</Code>
                {term.conflict.winner === term.instance_id ? (
                  <Status status="positive">{pageText(locale, 'This one', 'זה')}</Status>
                ) : (
                  <Status status="neutral">{pageText(locale, 'Not this one', 'לא זה')}</Status>
                )}
              </p>
            ) : null}
            {term.conflict.open && canEdit ? (
              <Button type="button" variant="outlined" onClick={() => onResolveConflict(term.conflict)}>
                <Scale size={14} aria-hidden="true" />
                {pageText(locale, 'Decide which clause governs', 'הכרעה איזה סעיף קובע')}
              </Button>
            ) : null}
          </div>
        ) : null}

        <div className="trd-term-meta">
          {taxonomy ? (
            <span className="trd-meta-item">
              <Status status={taxonomy.tone}>{taxonomy.label}</Status>
              <Prose as="span" className="trd-meta-note">{taxonomy.note}</Prose>
            </span>
          ) : null}
          {rank ? <span className="trd-chip-quiet">{rank.label}</span> : null}
          {effect ? <span className="trd-meta-note">{mechanismNote(effect.mechanism, locale)}</span> : null}
        </div>

        {scope.length > 0 ? (
          <dl className="trd-scope">
            {scope.map((line) => (
              <React.Fragment key={line.key}>
                <dt>{line.label}</dt>
                <dd><Name>{line.value}</Name></dd>
              </React.Fragment>
            ))}
          </dl>
        ) : null}

        {termWindow ? (
          <p className="trd-term-window">
            <span className="trd-card-label">{pageText(locale, 'Its own effective window', 'חלון התוקף שלו')}</span>
            <Figure>{termWindow}</Figure>
          </p>
        ) : null}

        {term.citations.length > 0 ? (
          <div className="trd-citations">
            <span className="trd-card-label">{pageText(locale, 'Where it says so', 'היכן זה כתוב')}</span>
            {term.citations.map((citation, index) => (
              <Button
                key={`${citation.clause_id || 'x'}-${index}`}
                type="button"
                className="trd-citation"
                onClick={() => onJumpToClause(citation.clause_id)}
                title={citation.quote || ''}
              >
                <Code>{citation.clause_id || pageText(locale, 'no clause', 'ללא סעיף')}</Code>
                {citation.page ? (
                  <Code>{pageText(locale, `p. ${citation.page}`, `עמ׳ ${citation.page}`)}</Code>
                ) : null}
                {citation.quote ? <Name className="trd-citation-quote">{citation.quote}</Name> : null}
              </Button>
            ))}
          </div>
        ) : (
          <p className="trd-field-hint">
            {term.notInDocument
              ? pageText(
                locale,
                'The reviewer recorded this term as agreed outside the document, so it has no citation.',
                'הסוקר רשם את המונח כמוסכם מחוץ למסמך, ולכן אין לו אסמכתא.',
              )
              : pageText(locale, 'This term carries no citation.', 'למונח הזה אין אסמכתא.')}
          </p>
        )}

        {term.notes ? <Prose className="trd-term-note">{term.notes}</Prose> : null}

        {term.state === 'rejected' && term.reason ? (
          <p className="trd-term-reason">
            <span className="trd-card-label">{pageText(locale, 'Reason for rejection', 'נימוק הדחייה')}</span>
            <Prose as="span">{term.reason}</Prose>
          </p>
        ) : null}

        {term.editedParams ? (
          <p className="trd-term-edited" role="note">
            <Pencil size={14} aria-hidden="true" />
            {pageText(
              locale,
              'A reviewer changed the extracted values. Both are kept: the effect above follows the edit.',
              'סוקר שינה את הערכים שחולצו. שניהם נשמרים: ההשפעה שלמעלה נגזרת מהתיקון.',
            )}
          </p>
        ) : null}

        <div className="trd-term-foot">
          <Button type="button" variant="outlined" onClick={() => setShowFields((open) => !open)} aria-expanded={showFields}>
            {showFields
              ? pageText(locale, 'Hide the extracted fields', 'הסתרת השדות שחולצו')
              : pageText(locale, 'Show the extracted fields', 'הצגת השדות שחולצו')}
          </Button>
          {canEdit ? (
            <div className="trd-term-actions">
              <Button type="button" onClick={() => onConfirm(term)} disabled={busy}>
                <Check size={14} aria-hidden="true" />
                {undecided
                  ? pageText(locale, 'Confirm', 'אישור')
                  : pageText(locale, 'Confirm again', 'אישור מחדש')}
              </Button>
              <Button type="button" variant="outlined" onClick={() => onEdit(term)} disabled={busy}>
                <Pencil size={14} aria-hidden="true" />
                {pageText(locale, 'Edit the values', 'תיקון הערכים')}
              </Button>
              <Button type="button" variant="outlined" onClick={() => onReject(term)} disabled={busy}>
                <Ban size={14} aria-hidden="true" />
                {pageText(locale, 'Reject', 'דחייה')}
              </Button>
            </div>
          ) : null}
        </div>

        {showFields ? (
          <div className="trd-fields">
            <p className="trd-field-hint">
              {pageText(
                locale,
                'The extracted fields, with the schema\'s own names. Nothing here is paraphrased.',
                'השדות שחולצו, בשמות של הסכימה עצמה. דבר כאן אינו מנוסח מחדש.',
              )}
            </p>
            <ParamTable params={params} locale={locale} />
            {term.editedParams ? (
              <>
                <p className="trd-field-hint">
                  {pageText(locale, 'What the document itself said, before the edit:', 'מה שהמסמך עצמו אמר, לפני התיקון:')}
                </p>
                <ParamTable params={term.params} locale={locale} />
              </>
            ) : null}
          </div>
        ) : null}
      </CardBody>
    </Card>
  );
}
