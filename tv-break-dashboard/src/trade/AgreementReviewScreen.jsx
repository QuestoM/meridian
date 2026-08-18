import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { EmptyState, ErrorState, LoadingState } from '../studio';
import { Button } from '../studio/actions';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Code, Name } from '../shell/bidi';
import { pageText } from '../shell/format';
import {
  acknowledgeClause, decideInstance, addInstance, approveAgreement, loadAgreement,
  loadProposal, markClausesSeen, refusalText, resolveConflict,
} from './trade-api';
import {
  buildClauses, buildConflicts, buildTerms, coverageOf, groupByMechanism,
  isUndecided, splitByStanding, termFilters,
} from './review-model';
import ReviewCoverageHeader from './ReviewCoverageHeader';
import ReviewDocumentPane from './ReviewDocumentPane';
import ReviewDialogs from './ReviewDialogs';
import ReviewProposalPane from './ReviewProposalPane';
import ReviewUnreadDocument, { useDocumentReading } from './ReviewUnreadDocument';
import './trade-agreements.css';
import './trade-review.css';
import './trade-lists.css';
import './trade-term-card.css';

// The review: the signed document beside what the engine proposes to do about it,
// and the gate that will not let either be approved until a person has been
// through all of it.
//
// This screen is the product's whole claim in one view. Everything else in the
// trade surface either feeds it or reports what it decided.
//
// WHAT IT REFUSES TO DO. It does not compute an effect — `effects` arrives from
// the compiler's own verdict over the CURRENT reviewed state, so every decision
// reloads the proposal and the sentences follow the edit. It does not enable
// approval from its own arithmetic — `gate.ready` is server truth and the button
// mirrors it. And it never reports a decision as saved before the server said so:
// a refusal keeps the dialog open with the store's own sentence in it.

export default function AgreementReviewScreen({
  agreementId, locale = 'he', notify = () => {}, canEdit = true, editRefusal = '',
  onClose, onOpenDetail,
}) {
  const [detail, setDetail] = useState(null);
  const [proposal, setProposal] = useState(null);
  const [documentId, setDocumentId] = useState('');
  const [error, setError] = useState(null);
  const [selectedClause, setSelectedClause] = useState('');
  const [filter, setFilter] = useState('undecided');
  const [action, setAction] = useState(null);
  const [actionError, setActionError] = useState('');
  const [busy, setBusy] = useState(false);
  const [marking, setMarking] = useState(false);
  const [approving, setApproving] = useState(false);
  // A document with no extraction yet is a STAGE, not a failure: the reading
  // has not been run. ReviewUnreadDocument offers it and watches the job.
  const [unread, setUnread] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const reload = useCallback(() => setReloadKey((key) => key + 1), []);
  const Back = locale === 'he' ? ChevronRight : ChevronLeft;

  useEffect(() => {
    let alive = true;
    setError(null);
    setDetail(null);
    loadAgreement(agreementId).then(
      (payload) => {
        if (!alive) return;
        setDetail(payload);
        const documents = (payload.agreement && payload.agreement.documents) || [];
        setDocumentId((current) => (
          current && documents.some((d) => d.document_id === current)
            ? current
            : (documents.length > 0 ? documents[0].document_id : '')
        ));
      },
      (failure) => { if (alive) setError(failure); },
    );
    return () => { alive = false; };
  }, [agreementId, reloadKey]);

  useEffect(() => {
    if (!documentId) return undefined;
    let alive = true;
    setProposal(null);
    setError(null);
    setUnread(false);
    loadProposal(agreementId, documentId).then(
      (payload) => { if (alive) setProposal(payload); },
      (failure) => {
        if (!alive) return;
        // 404 on the proposal means this document has not been read yet. That
        // is the stage before a review exists, and it has its own screen with
        // the action that ends it - not an error with the server's words in it.
        if (failure && failure.status === 404) setUnread(true);
        else setError(failure);
      },
    );
    return () => { alive = false; };
  }, [agreementId, documentId, reloadKey]);

  const { reading, failure: readingFailure, run: runReading } = useDocumentReading({
    agreementId, documentId, locale, notify, onRead: reload,
  });

  const terms = useMemo(() => (proposal ? buildTerms(proposal) : []), [proposal]);
  const clauses = useMemo(() => (proposal ? buildClauses(proposal) : []), [proposal]);
  const conflicts = useMemo(() => (proposal ? buildConflicts(proposal) : []), [proposal]);
  const coverage = useMemo(() => (proposal ? coverageOf(proposal) : null), [proposal]);
  const counts = useMemo(
    () => termFilters(terms, conflicts, clauses),
    [terms, conflicts, clauses],
  );

  // Which terms the selected clause is evidence for. Selecting a clause does not
  // filter the list — it highlights, because a reviewer who lost the rest of the
  // agreement to see one clause's terms has lost the context the review is for.
  const clauseTermIds = useMemo(() => {
    const clause = clauses.find((entry) => entry.clause_id === selectedClause);
    return new Set(clause ? clause.instanceIds : []);
  }, [clauses, selectedClause]);

  const shown = useMemo(() => {
    if (filter === 'undecided') return terms.filter(isUndecided);
    if (filter === 'inert') return terms.filter((t) => t.effect && t.effect.mechanism === 'inert');
    if (filter === 'incomplete') {
      return terms.filter((t) => t.missing.length > 0 || (t.effect && t.effect.incomplete));
    }
    if (filter === 'conflicts') return terms.filter((t) => t.conflict && t.conflict.open);
    return terms;
  }, [terms, filter]);

  // The proposals a person must decide, and the readings that only suggest a
  // term might live in a clause. Only the first kind is grouped and counted:
  // the second holds nothing shut and lives in its own folded list.
  const split = useMemo(() => splitByStanding(shown), [shown]);
  const groups = useMemo(() => groupByMechanism(split.proposals), [split]);
  const unacknowledged = useMemo(
    () => clauses.filter((clause) => clause.disposition === 'unmapped' && !clause.acknowledged),
    [clauses],
  );

  // Selecting a clause records that a reviewer had it on screen. It is the same
  // act as reading it, which is exactly what the gate is asking about.
  async function selectClause(clause) {
    setSelectedClause(clause.clause_id);
    if (!clause.seen && canEdit) {
      try {
        await markClausesSeen(agreementId, documentId, [clause.clause_id]);
        reload();
      } catch (failure) {
        notify(
          `The clause could not be marked as read. ${refusalText(failure, 'en')}`,
          `לא ניתן היה לסמן את הסעיף כנקרא. ${refusalText(failure, 'he')}`,
        );
      }
    }
  }

  async function markSeen(clauseIds) {
    setMarking(true);
    try {
      await markClausesSeen(agreementId, documentId, clauseIds);
      reload();
    } catch (failure) {
      notify(
        `Those clauses could not be marked as read. ${refusalText(failure, 'en')}`,
        `לא ניתן היה לסמן את הסעיפים כנקראו. ${refusalText(failure, 'he')}`,
      );
    } finally {
      setMarking(false);
    }
  }

  async function confirm(term) {
    setBusy(true);
    try {
      await decideInstance(agreementId, documentId, term.instance_id, { verdict: 'confirmed' });
      reload();
      notify(
        `${term.instance_id} confirmed.`,
        `${term.instance_id} אושר.`,
      );
    } catch (failure) {
      notify(
        `The term could not be confirmed. ${refusalText(failure, 'en')}`,
        `לא ניתן היה לאשר את המונח. ${refusalText(failure, 'he')}`,
      );
    } finally {
      setBusy(false);
    }
  }

  // A reading with no values in it holds nothing shut, so a reviewer who
  // recognises a real term in one says so here — and from that moment it is an
  // ordinary proposal that must be decided before this agreement can be
  // approved. There is no way back other than rejecting it with a reason,
  // which is what the ordinary decision path already records.
  async function promote(term) {
    setBusy(true);
    try {
      await promoteInstance(agreementId, documentId, term.instance_id);
      reload();
      notify(
        `${term.instance_id} moved into the proposals; it now needs a decision.`,
        `${term.instance_id} הועבר להצעות, וכעת הוא דורש הכרעה.`,
      );
    } catch (failure) {
      notify(
        `The reading could not be moved. ${refusalText(failure, 'en')}`,
        `לא ניתן היה להעביר את הקריאה. ${refusalText(failure, 'he')}`,
      );
    } finally {
      setBusy(false);
    }
  }

  async function submitAction(payload) {
    setBusy(true);
    setActionError('');
    try {
      if (payload.kind === 'reject' || payload.kind === 'edit') {
        await decideInstance(agreementId, documentId, action.term.instance_id, {
          verdict: payload.verdict,
          edited_params: payload.edited_params,
          reason: payload.reason,
        });
      }
      if (payload.kind === 'add') {
        await addInstance(agreementId, documentId, {
          term_id: payload.term_id,
          params: payload.params,
          clause_id: payload.clause_id,
          quote: payload.quote,
          not_in_document: payload.not_in_document,
          note: payload.note,
        });
      }
      if (payload.kind === 'acknowledge') {
        await acknowledgeClause(agreementId, documentId, action.clause.clause_id, payload.note);
      }
      if (payload.kind === 'conflict') {
        await resolveConflict(agreementId, documentId, action.conflict.conflict_id, {
          winner_instance_id: payload.winner_instance_id,
          note: payload.note,
        });
      }
      setAction(null);
      reload();
      notify(
        'The review was updated.',
        'הסקירה עודכנה.',
      );
    } catch (failure) {
      setActionError(refusalText(failure, locale));
    } finally {
      setBusy(false);
    }
  }

  async function approve() {
    setApproving(true);
    try {
      const result = await approveAgreement(agreementId);
      const skipped = (result.compiled && result.compiled.skipped) || [];
      notify(
        `Approved as version ${result.version.version_id}. ${skipped.length} terms were not compiled.`,
        `אושר כגרסה ${result.version.version_id}. ${skipped.length} מונחים לא נקמפלו.`,
      );
      if (onOpenDetail) onOpenDetail();
    } catch (failure) {
      notify(
        `The agreement could not be approved. ${refusalText(failure, 'en')}`,
        `לא ניתן היה לאשר את ההסכם. ${refusalText(failure, 'he')}`,
      );
    } finally {
      setApproving(false);
    }
  }

  const head = detail ? detail.agreement : null;
  const documents = head ? head.documents || [] : [];

  if (error && !proposal) {
    return (
      <ErrorState
        title={pageText(locale, 'The review could not be opened', 'לא ניתן היה לפתוח את הסקירה')}
        description={refusalText(error, locale)}
        action={(
          <div className="trd-header-actions">
            <Button type="button" variant="outlined" onClick={onClose}>
              {pageText(locale, 'Back to the agreements', 'חזרה לרשימת ההסכמים')}
            </Button>
            <Button type="button" variant="contained" onClick={reload}>{pageText(locale, 'Try again', 'נסו שוב')}</Button>
          </div>
        )}
      />
    );
  }

  // The document is attached but never read: a stage, not a failure.
  if (unread && !proposal) {
    return (
      <ReviewUnreadDocument
        document={documents.find((d) => d.document_id === documentId)}
        locale={locale}
        reading={reading}
        failure={readingFailure}
        onRun={runReading}
        onClose={onClose}
        canEdit={canEdit}
        editRefusal={editRefusal}
      />
    );
  }

  if (!detail || (documentId && !proposal)) {
    return (
      <LoadingState
        title={pageText(locale, 'Opening the review', 'פותח את הסקירה')}
        description={pageText(
          locale,
          'The document, the proposed terms and the completeness gate are read together, so nothing is shown against a stale gate.',
          'המסמך, המונחים המוצעים ושער השלמות נקראים יחד, כדי שדבר לא יוצג מול שער לא עדכני.',
        )}
      />
    );
  }

  if (!documentId) {
    return (
      <EmptyState
        title={pageText(locale, 'This agreement has no document', 'להסכם הזה אין מסמך')}
        description={pageText(
          locale,
          'There is nothing to review until a signed document is attached and read.',
          'אין מה לסקור עד שיצורף מסמך חתום וייקרא.',
        )}
        action={(
          <Button type="button" variant="outlined" onClick={onClose}>
            {pageText(locale, 'Back to the agreements', 'חזרה לרשימת ההסכמים')}
          </Button>
        )}
      />
    );
  }

  return (
    <section className="trd-review" aria-label={pageText(locale, 'Agreement review', 'סקירת הסכם')}>
      <header className="trd-review-head">
        <div className="trd-review-title">
          <Button type="button" variant="outlined" onClick={onClose}>
            <Back size={16} aria-hidden="true" />
            {pageText(locale, 'All agreements', 'כל ההסכמים')}
          </Button>
          <div>
            <h2><Name>{head.title}</Name></h2>
            <p className="trd-review-sub">
              <Code>{agreementId}</Code>
              {documents.length > 1 ? null : (
                <Code>{documents[0] ? documents[0].filename : documentId}</Code>
              )}
            </p>
          </div>
        </div>
        <div className="trd-header-actions">
          {documents.length > 1 ? (
            <div className="trd-doc-switch" role="group" aria-label={pageText(locale, 'Documents', 'מסמכים')}>
              {documents.map((document) => (
                <Button
                  key={document.document_id}
                  type="button"
                  variant={document.document_id === documentId ? 'contained' : 'outlined'}
                  onClick={() => setDocumentId(document.document_id)}
                >
                  <Name>{document.filename}</Name>
                </Button>
              ))}
            </div>
          ) : null}
          <Button type="button" variant="outlined" onClick={onOpenDetail}>
            {pageText(locale, 'The agreement record', 'רשומת ההסכם')}
          </Button>
        </div>
      </header>

      <ReviewCoverageHeader
        coverage={coverage}
        locale={locale}
        canEdit={canEdit}
        editRefusal={editRefusal}
        approving={approving}
        onApprove={approve}
      />

      <div className="trd-review-body">
        <ReviewDocumentPane
          agreementId={agreementId}
          documentId={documentId}
          clauses={clauses}
          locale={locale}
          selected={selectedClause}
          onSelect={selectClause}
          onMarkSeen={markSeen}
          canEdit={canEdit}
          marking={marking}
        />

        <ReviewProposalPane
          locale={locale}
          canEdit={canEdit}
          busy={busy}
          filter={filter}
          counts={counts}
          groups={groups}
          shown={split.proposals}
          interpretations={split.interpretations}
          onPromote={promote}
          conflicts={conflicts}
          unacknowledged={unacknowledged}
          selectedClause={selectedClause}
          clauseTermIds={clauseTermIds}
          onFilter={setFilter}
          onSelectClause={setSelectedClause}
          onConfirm={confirm}
          on={{
            add: () => setAction({ kind: 'add' }),
            acknowledge: (clause) => setAction({ kind: 'acknowledge', clause }),
            edit: (target) => setAction({ kind: 'edit', term: target }),
            reject: (target) => setAction({ kind: 'reject', term: target }),
            resolveConflict: (conflict) => setAction({ kind: 'conflict', conflict }),
          }}
        />
      </div>

      {action ? (
        <ReviewDialogs
          key={`${action.kind}-${(action.term && action.term.instance_id) || (action.clause && action.clause.clause_id) || (action.conflict && action.conflict.conflict_id) || 'new'}`}
          action={action}
          locale={locale}
          busy={busy}
          error={actionError}
          onClose={() => { setAction(null); setActionError(''); }}
          onSubmit={submitAction}
        />
      ) : null}
    </section>
  );
}
