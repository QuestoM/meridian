import React, { useCallback, useEffect, useState } from 'react';
import { Card, CardBody, ErrorState, LoadingState, Status } from '../studio';
import { Button } from '../studio/actions';
import {
  ChevronLeft, ChevronRight, FileCheck2, FileSearch, FileText, Stamp,
} from 'lucide-react';
import { Code, Figure, Name, Prose } from '../shell/bidi';
import { formatDay, formatSpan, formatStamp } from '../shell/dates';
import { formatNumber, pageText } from '../shell/format';
import {
  counterpartyKind, counterpartyKindOf, counterpartyName, levelLabel, openEndedLabel,
  statusLabel, statusTone, windowOf,
} from './trade-vocabulary';
import {
  loadAgreement, loadObligations, loadProposal, refusalText, simulateAgreement,
} from './trade-api';
import { AgreementTermsByFamily, BoundRules } from './AgreementTermsAndRules';
import ObligationsBoard from './ObligationsBoard';
import SimulationPanel from './SimulationPanel';
import './trade-agreements.css';
import './trade-review.css';
import './trade-record.css';
import './trade-lists.css';
import './trade-term-card.css';

// The agreement record: what was signed, what was approved, what is live, what it
// committed the channel to, and what it would do to real money.
//
// The order is the order a commercial director asks the questions in. Identity and
// document first, because provenance decides whether the rest can be trusted at
// all — the document's checksum is on the screen for that reason. Then the
// approval, because an approved version is the only thing that can bind. Then the
// terms and the live rules side by side. Then the standing of every commitment.
// Then, last, the money the deal would move.
//
// FOUR READS, FOUR STATES. The record, the reviewed terms, the commitment standing
// and the simulation are separate server reads, and one failing does not blank the
// others: each names its own failure where it would have rendered. The simulation
// is the only one that is not read on open, because it is a computation over the
// whole delivery ledger and running it is the operator's decision.

function Line({ label, children }) {
  return (
    <div className="trd-line">
      <span className="trd-card-label">{label}</span>
      <span className="trd-line-value">{children}</span>
    </div>
  );
}

function DocumentRow({ document, locale, onOpenReview }) {
  return (
    <li className="trd-doc-row">
      <span className="trd-doc-main">
        <FileText size={15} aria-hidden="true" />
        <Name className="trd-doc-name">{document.filename}</Name>
        <Code className="trd-id-chip">{document.document_id}</Code>
      </span>
      <span className="trd-doc-meta">
        <span className="trd-chip-quiet">
          {document.ingest_route === 'scanned'
            ? pageText(locale, 'Read from a scan', 'נקרא מסריקה')
            : pageText(locale, 'Read from digital text', 'נקרא מטקסט דיגיטלי')}
        </span>
        <Figure>
          {pageText(
            locale,
            `${formatNumber(Math.round(Number(document.bytes || 0) / 1024), locale)} KB`,
            `${formatNumber(Math.round(Number(document.bytes || 0) / 1024), locale)} ק״ב`,
          )}
        </Figure>
        <Figure>{formatStamp(document.attached_at)}</Figure>
      </span>
      {/* The checksum is not decoration. It is what makes an approved version name
          the exact bytes that were read, so a document swapped afterwards cannot
          pass as the one somebody signed off. */}
      <Code className="trd-sha">{String(document.sha256 || '').slice(0, 16)}</Code>
      <Button type="button" variant="outlined" onClick={onOpenReview}>
        <FileSearch size={14} aria-hidden="true" />
        {pageText(locale, 'Open the review', 'פתיחת הסקירה')}
      </Button>
    </li>
  );
}

function VersionCard({ version, current, locale }) {
  const counts = version.counts || {};
  return (
    <Card className="trd-version" data-current={current ? 'true' : undefined}>
      <CardBody>
        <div className="trd-ob-head">
          <div>
            <h5>
              <Code>{version.version_id}</Code>
            </h5>
            <span className="trd-chip-quiet">
              {pageText(
                locale,
                `version ${formatNumber(version.seq, locale)}`,
                `גרסה ${formatNumber(version.seq, locale)}`,
              )}
            </span>
          </div>
          {current ? (
            <Status status="positive">{pageText(locale, 'The governing version', 'הגרסה הקובעת')}</Status>
          ) : (
            <Status status="neutral">{pageText(locale, 'Superseded', 'הוחלפה')}</Status>
          )}
        </div>
        <Line label={pageText(locale, 'Approved by', 'אושר על ידי')}>
          <Name>{version.actor}</Name>
        </Line>
        <Line label={pageText(locale, 'Approved at', 'אושר בתאריך')}>
          <Figure>{formatStamp(version.created_at)}</Figure>
        </Line>
        {version.note ? (
          <Line label={pageText(locale, 'Approval note', 'הערת אישור')}>
            <Prose as="span">{version.note}</Prose>
          </Line>
        ) : null}
        <dl className="trd-sim-block">
          <dt>{pageText(locale, 'Terms approved', 'מונחים שאושרו')}</dt>
          <dd><Figure>{formatNumber(counts.approved_terms, locale)}</Figure></dd>
          <dt>{pageText(locale, 'Terms rejected', 'מונחים שנדחו')}</dt>
          <dd><Figure>{formatNumber(counts.rejected_terms, locale)}</Figure></dd>
          <dt>{pageText(locale, 'Clauses acknowledged with no term', 'סעיפים שאושרו ללא מונח')}</dt>
          <dd><Figure>{formatNumber(counts.acknowledged_unsupported, locale)}</Figure></dd>
          <dt>{pageText(locale, 'Conflicts settled', 'סתירות שהוכרעו')}</dt>
          <dd><Figure>{formatNumber(counts.conflicts, locale)}</Figure></dd>
        </dl>
        {(version.documents || []).map((document) => (
          <Line key={document.document_id} label={pageText(locale, 'Document it froze', 'המסמך שהוקפא')}>
            <Name>{document.filename}</Name>
            <Code className="trd-sha">{String(document.sha256 || '').slice(0, 16)}</Code>
          </Line>
        ))}
      </CardBody>
    </Card>
  );
}

export default function AgreementDetailScreen({
  agreementId, locale = 'he', notify = () => {}, onClose, onOpenReview,
}) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);
  const [effects, setEffects] = useState(null);
  const [effectsError, setEffectsError] = useState('');
  const [obligations, setObligations] = useState(null);
  const [obligationsError, setObligationsError] = useState('');
  const [simulation, setSimulation] = useState(null);
  const [simulationError, setSimulationError] = useState('');
  const [simulating, setSimulating] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const reload = useCallback(() => setReloadKey((key) => key + 1), []);
  const Back = locale === 'he' ? ChevronRight : ChevronLeft;

  useEffect(() => {
    let alive = true;
    setError(null);
    setDetail(null);
    loadAgreement(agreementId).then(
      (payload) => { if (alive) setDetail(payload); },
      (failure) => { if (alive) setError(failure); },
    );
    return () => { alive = false; };
  }, [agreementId, reloadKey]);

  // The reviewed terms of the first attached document. This is what the agreement
  // holds; for an approved agreement it is frozen, which is why it can stand in
  // for the approved termset without a second route.
  useEffect(() => {
    if (!detail) return undefined;
    const documents = (detail.agreement && detail.agreement.documents) || [];
    if (documents.length === 0) { setEffects({ terms: [] }); return undefined; }
    let alive = true;
    setEffectsError('');
    loadProposal(agreementId, documents[0].document_id).then(
      (payload) => { if (alive) setEffects(payload.effects || { terms: [] }); },
      (failure) => { if (alive) setEffectsError(refusalText(failure, locale)); },
    );
    return () => { alive = false; };
  }, [agreementId, detail, locale]);

  useEffect(() => {
    let alive = true;
    setObligationsError('');
    setObligations(null);
    loadObligations(agreementId).then(
      (payload) => { if (alive) setObligations(payload); },
      (failure) => { if (alive) setObligationsError(refusalText(failure, locale)); },
    );
    return () => { alive = false; };
  }, [agreementId, locale, reloadKey]);

  async function runSimulation(windowRange) {
    setSimulating(true);
    setSimulationError('');
    try {
      const payload = await simulateAgreement(agreementId, windowRange ? { window: windowRange } : {});
      setSimulation(payload);
    } catch (failure) {
      setSimulationError(refusalText(failure, locale));
      notify(
        `The simulation could not be run. ${refusalText(failure, 'en')}`,
        `לא ניתן היה להריץ את הסימולציה. ${refusalText(failure, 'he')}`,
      );
    } finally {
      setSimulating(false);
    }
  }

  if (error) {
    return (
      <ErrorState
        title={pageText(locale, 'The agreement could not be read', 'לא ניתן היה לקרוא את ההסכם')}
        description={refusalText(error, locale)}
        action={(
          <div className="trd-header-actions">
            <Button type="button" variant="outlined" onClick={onClose}>
              {pageText(locale, 'Back to the agreements', 'חזרה לרשימת ההסכמים')}
            </Button>
            <Button type="button" onClick={reload}>{pageText(locale, 'Try again', 'נסו שוב')}</Button>
          </div>
        )}
      />
    );
  }

  if (!detail) {
    return (
      <LoadingState
        title={pageText(locale, 'Reading the agreement', 'קורא את ההסכם')}
        description={pageText(
          locale,
          'The record, its documents, its approved versions and the live rules it owns are read together.',
          'הרשומה, המסמכים שלה, הגרסאות המאושרות והכללים הפעילים שהיא מחזיקה נקראים יחד.',
        )}
      />
    );
  }

  const head = detail.agreement;
  const documents = head.documents || [];
  const versions = detail.versions || [];
  const span = windowOf(head.window);
  const party = counterpartyName(head.counterparty);

  return (
    <section className="trd-detail" aria-label={pageText(locale, 'Agreement record', 'רשומת הסכם')}>
      <header className="trd-review-head">
        <div className="trd-review-title">
          <Button type="button" variant="outlined" onClick={onClose}>
            <Back size={16} aria-hidden="true" />
            {pageText(locale, 'All agreements', 'כל ההסכמים')}
          </Button>
          <div>
            <h2><Name>{head.title}</Name></h2>
            <p className="trd-review-sub">
              <Code>{head.agreement_id}</Code>
              <Status status={statusTone(head.status)}>{statusLabel(head.status, locale)}</Status>
            </p>
          </div>
        </div>
        {documents.length > 0 ? (
          <div className="trd-header-actions">
            <Button type="button" onClick={onOpenReview}>
              <FileSearch size={14} aria-hidden="true" />
              {pageText(locale, 'Open the review', 'פתיחת הסקירה')}
            </Button>
          </div>
        ) : null}
      </header>

      <Card className="trd-identity">
        <CardBody>
          <div className="trd-identity-grid">
            <Line label={pageText(locale, 'Agreement level', 'רמת ההסכם')}>
              {levelLabel(head.level, locale)}
            </Line>
            {party ? (
              <Line label={counterpartyKind(counterpartyKindOf(head.counterparty), locale)}>
                <Name>{party}</Name>
              </Line>
            ) : null}
            <Line label={pageText(locale, 'Effective window', 'תקופת תוקף')}>
              {span.openEnded ? (
                <>
                  <Figure>{formatDay(span.from)}</Figure>
                  <span className="trd-chip-quiet">{openEndedLabel(locale)}</span>
                </>
              ) : (
                <Figure>{formatSpan(span.from, span.to, locale)}</Figure>
              )}
            </Line>
            <Line label={pageText(locale, 'Created', 'נוצר')}>
              <Figure>{formatStamp(head.created_at)}</Figure>
              <Name>{head.created_by}</Name>
            </Line>
            {head.parent_agreement_id ? (
              <Line label={pageText(locale, 'Under the agreement', 'תחת ההסכם')}>
                <Code>{head.parent_agreement_id}</Code>
              </Line>
            ) : null}
            <Line label={pageText(locale, 'Governing version', 'הגרסה הקובעת')}>
              {head.current_version_id
                ? <Code>{head.current_version_id}</Code>
                : (
                  <span className="trd-unknown">
                    {pageText(locale, 'none approved yet', 'טרם אושרה גרסה')}
                  </span>
                )}
            </Line>
          </div>
          {head.note ? <Prose className="trd-identity-note">{head.note}</Prose> : null}
        </CardBody>
      </Card>

      <section aria-label={pageText(locale, 'Documents', 'מסמכים')}>
        <div className="trd-pane-head">
          <h4>
            <FileText size={16} aria-hidden="true" />
            {pageText(locale, 'Documents', 'מסמכים')}
          </h4>
        </div>
        {documents.length === 0 ? (
          <p className="trd-field-hint">
            {pageText(locale, 'No document is attached, so there is nothing to review.', 'אין מסמך מצורף, ולכן אין מה לסקור.')}
          </p>
        ) : (
          <ul className="trd-doc-list">
            {documents.map((document) => (
              <DocumentRow
                key={document.document_id}
                document={document}
                locale={locale}
                onOpenReview={onOpenReview}
              />
            ))}
          </ul>
        )}
      </section>

      <section aria-label={pageText(locale, 'Approved versions', 'גרסאות מאושרות')}>
        <div className="trd-pane-head">
          <h4>
            <Stamp size={16} aria-hidden="true" />
            {pageText(locale, 'Approved versions', 'גרסאות מאושרות')}
          </h4>
        </div>
        {versions.length === 0 ? (
          <p className="trd-field-hint">
            {pageText(
              locale,
              'No version has been approved, so this agreement changes nothing yet. Approval is what freezes a version and lets the engine act.',
              'לא אושרה גרסה, ולכן ההסכם הזה אינו משנה דבר בשלב הזה. האישור הוא מה שמקפיא גרסה ומאפשר למנוע לפעול.',
            )}
          </p>
        ) : (
          <div className="trd-version-grid">
            {versions.map((version) => (
              <VersionCard
                key={version.version_id}
                version={version}
                current={version.version_id === head.current_version_id}
                locale={locale}
              />
            ))}
          </div>
        )}
      </section>

      {effectsError ? (
        <ErrorState
          title={pageText(locale, 'The terms could not be read', 'לא ניתן היה לקרוא את המונחים')}
          description={effectsError}
        />
      ) : effects === null ? (
        <LoadingState title={pageText(locale, 'Reading the terms', 'קורא את המונחים')} />
      ) : (
        <AgreementTermsByFamily effects={effects} locale={locale} />
      )}

      <BoundRules boundRules={detail.bound_rules} locale={locale} />

      <ObligationsBoard
        payload={obligations}
        error={obligationsError}
        locale={locale}
        onRetry={(
          <Button type="button" onClick={reload}>{pageText(locale, 'Try again', 'נסו שוב')}</Button>
        )}
      />

      {/* A simulation writes nothing, so it is offered to a read-only account
          too: refusing a read-only operator the ability to ask what a deal would
          do would be a permission with no consequence to protect. */}
      <SimulationPanel
        payload={simulation}
        error={simulationError}
        locale={locale}
        busy={simulating}
        canRun
        onRun={runSimulation}
      />

      <Card dense className="trd-boundary">
        <CardBody>
          <FileCheck2 size={16} aria-hidden="true" />
          <p>
            {pageText(
              locale,
              'Every live rule above can be traced back to the clause that put it there. Nothing in the engine attributed to this agreement exists without a citation in a document with a pinned checksum.',
              'כל כלל פעיל שלמעלה ניתן לייחוס חזרה לסעיף שהוליד אותו. דבר במנוע שמיוחס להסכם הזה אינו קיים בלי אסמכתא במסמך שטביעת האצבע שלו מקובעת.',
            )}
          </p>
        </CardBody>
      </Card>
    </section>
  );
}
