// One reviewable proposal, assembled from the four things the server sends.
//
// The proposal endpoint returns four separate objects and each holds a different
// half of the same term:
//
//   extraction  what the document said     params, citations, confidence, missing
//   review      what the reviewer decided  state, edited params, reason, additions
//   effects     what it WILL DO            mechanism, sentence, will-not-act reasons
//   gate        why approval is refused    blockers by kind, with ids
//
// A screen that reaches into all four at every render re-derives the join
// dozens of times and gets it subtly wrong once. So the join happens here,
// once, as plain data with no framework in it.
//
// TWO JOIN FACTS THAT ARE EASY TO GET WRONG, both load-bearing.
//
// `effects.terms` is computed over the EFFECTIVE termset: rejected instances are
// absent from it and reviewer-added ones are present. So a term with no effect
// entry is not an error and must not render as one — it is a rejected term, and
// it has no mechanism because nothing will act on it.
//
// `clauses_seen` and `unmapped_acks` are objects keyed by clause id, not arrays.
// Reading them with .includes() silently reports every clause unseen, which
// would make the coverage header lie in the direction that lets an unread
// agreement look ready.

import { MECHANISM_ORDER } from './trade-vocabulary';

const PROPOSED = 'proposed';
const REJECTED = 'rejected';

function asObject(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

// One row per term the reviewer must decide on, extraction and reviewer
// additions together, each carrying its own effect when it has one.
export function buildTerms(proposal) {
  const extraction = asObject(proposal.extraction);
  const review = asObject(proposal.review);
  const effects = asObject(proposal.effects);
  const states = asObject(review.instances);
  const effectById = new Map(asArray(effects.terms).map((term) => [term.instance_id, term]));
  // Where each reading sits — a proposal to decide, or an interpretation to
  // consider. The server decides it (kairos.trade.standing) so this screen and
  // the approval gate can never disagree about which list a term belongs in.
  const standings = asObject(proposal.standings);
  const conflictByInstance = new Map();
  Object.entries(asObject(review.conflicts)).forEach(([conflictId, conflict]) => {
    asArray(conflict.instances).forEach((instanceId) => {
      conflictByInstance.set(instanceId, { conflict_id: conflictId, ...conflict });
    });
  });

  const rows = asArray(extraction.instances).map((instance) => {
    const entry = asObject(states[instance.instance_id]);
    const state = String(entry.state || PROPOSED);
    return {
      instance_id: instance.instance_id,
      term_id: instance.term_id,
      params: instance.params || {},
      editedParams: entry.edited_params || null,
      scope: entry.edited_scope || instance.scope || {},
      window: entry.edited_window || instance.window || {},
      citations: asArray(instance.citations),
      confidence: instance.confidence || '',
      missing: asArray(instance.missing),
      notes: instance.notes || '',
      state,
      decidedBy: entry.by || '',
      decidedAt: entry.at || '',
      reason: entry.reason || '',
      addedByReviewer: false,
      notInDocument: false,
      effect: effectById.get(instance.instance_id) || null,
      conflict: conflictByInstance.get(instance.instance_id) || null,
      standing: asObject(standings[instance.instance_id]).standing || 'confident',
      standingReason: asObject(standings[instance.instance_id]),
    };
  });

  const added = asArray(review.reviewer_added).map((instance) => ({
    instance_id: instance.instance_id,
    term_id: instance.term_id,
    params: instance.params || {},
    editedParams: null,
    scope: instance.scope || {},
    window: instance.window || {},
    citations: instance.clause_id
      ? [{ clause_id: instance.clause_id, quote: instance.quote || '' }]
      : [],
    confidence: '',
    missing: asArray(instance.missing),
    notes: instance.note || '',
    state: 'reviewer_added',
    decidedBy: instance.added_by || '',
    decidedAt: instance.added_at || '',
    reason: '',
    addedByReviewer: true,
    notInDocument: Boolean(instance.not_in_document),
    effect: effectById.get(instance.instance_id) || null,
    conflict: conflictByInstance.get(instance.instance_id) || null,
    // A term a reviewer wrote is a proposal by definition: a person put the
    // values in it.
    standing: 'confident',
    standingReason: {},
  }));

  return rows.concat(added);
}

// One row per clause in the document, with how the extraction disposed of it,
// whether a reviewer has had it on screen, and which terms cite it.
export function buildClauses(proposal) {
  const extraction = asObject(proposal.extraction);
  const review = asObject(proposal.review);
  const seen = asObject(review.clauses_seen);
  const acks = asObject(review.unmapped_acks);
  const dispositions = new Map(
    asArray(extraction.dispositions).map((entry) => [entry.clause_id, entry]),
  );
  const citedBy = new Map();
  buildTerms(proposal).forEach((term) => {
    term.citations.forEach((citation) => {
      if (!citation.clause_id) return;
      const list = citedBy.get(citation.clause_id) || [];
      list.push(term.instance_id);
      citedBy.set(citation.clause_id, list);
    });
  });

  return asArray(extraction.clauses).map((clause) => {
    const disposition = dispositions.get(clause.clause_id) || {};
    return {
      clause_id: clause.clause_id,
      text: clause.text || '',
      heading: clause.heading || '',
      pages: asArray(clause.pages),
      isTable: Boolean(clause.is_table),
      disposition: disposition.disposition || 'unmapped',
      irrelevantClass: disposition.irrelevant_class || '',
      dispositionReason: disposition.reason || '',
      instanceIds: asArray(disposition.instance_ids).length
        ? asArray(disposition.instance_ids)
        : (citedBy.get(clause.clause_id) || []),
      seen: Object.prototype.hasOwnProperty.call(seen, clause.clause_id),
      seenAt: seen[clause.clause_id] ? seen[clause.clause_id].at : '',
      acknowledged: Object.prototype.hasOwnProperty.call(acks, clause.clause_id),
      acknowledgeNote: acks[clause.clause_id] ? acks[clause.clause_id].note : '',
    };
  });
}

// A conflict is open until the resolver's rule or a human settles it. Anything
// else is an unsettled contradiction between two clauses of the same agreement,
// and it blocks approval.
const SETTLED = new Set(['resolved_by_rule', 'resolved_by_human']);

export function buildConflicts(proposal) {
  const review = asObject(proposal.review);
  return Object.entries(asObject(review.conflicts)).map(([conflictId, conflict]) => ({
    conflict_id: conflictId,
    instances: asArray(conflict.instances),
    contested: conflict.contested || '',
    resolution: conflict.resolution || '',
    winner: conflict.winner || '',
    rule: conflict.rule || '',
    explanationHe: conflict.explanation_he || '',
    note: conflict.note || '',
    resolvedBy: conflict.resolved_by || '',
    open: !SETTLED.has(String(conflict.resolution || '')),
  }));
}

// The coverage a reviewer is held to, from the gate's own numbers where the gate
// supplies them. The gate is the authority on readiness; this only reshapes what
// it said so the header can print it without arithmetic of its own.
export function coverageOf(proposal) {
  const gate = asObject(proposal.gate);
  const clauses = buildClauses(proposal);
  const dispositions = asObject(gate.dispositions);
  return {
    clausesTotal: Number(gate.clauses_total ?? clauses.length),
    clausesSeen: Number(gate.clauses_seen ?? clauses.filter((c) => c.seen).length),
    mapped: Number(dispositions.mapped ?? 0),
    irrelevant: Number(dispositions.irrelevant ?? 0),
    unmapped: Number(dispositions.unmapped ?? 0),
    unmappedAcknowledged: Number(gate.unmapped_acknowledged ?? 0),
    instancesTotal: Number(gate.instances_total ?? 0),
    instancesDecided: Number(gate.instances_decided ?? 0),
    reviewerAdded: Number(gate.reviewer_added ?? 0),
    conflictsOpen: Number(gate.conflicts_open ?? 0),
    ready: Boolean(gate.ready),
    blockers: asArray(gate.blockers),
  };
}

// Terms grouped by what they will do, in the order that puts the dangerous
// reading first: a term that will NOT act, then the ones that refuse placements,
// then money, then measurement, then the merely recorded. A rejected term has no
// mechanism and is grouped under its own heading at the end.
export const REJECTED_GROUP = 'rejected';

export function groupByMechanism(terms) {
  const groups = new Map();
  const order = ['inert', ...MECHANISM_ORDER.filter((m) => m !== 'inert'), REJECTED_GROUP];
  order.forEach((key) => groups.set(key, []));
  terms.forEach((term) => {
    const key = term.state === REJECTED
      ? REJECTED_GROUP
      : (term.effect ? String(term.effect.mechanism || 'records') : REJECTED_GROUP);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(term);
  });
  return order
    .map((key) => ({ key, terms: groups.get(key) || [] }))
    .filter((group) => group.terms.length > 0);
}

export function isUndecided(term) {
  return term.state === PROPOSED;
}

export function termFilters(terms, conflicts, clauses) {
  return {
    all: terms.length,
    undecided: terms.filter(isUndecided).length,
    inert: terms.filter((t) => t.effect && t.effect.mechanism === 'inert').length,
    incomplete: terms.filter((t) => t.missing.length > 0 || (t.effect && t.effect.incomplete)).length,
    conflicts: conflicts.filter((c) => c.open).length,
    unmapped: clauses.filter((c) => c.disposition === 'unmapped' && !c.acknowledged).length,
  };
}


// The two lists a reading can be in. Splitting them is what keeps the list a
// person approves short enough to actually work through: measured on the
// corpus, 16 of 228 readings carry no values at all, and setting those aside
// raises the share of the main list that is correct from 66.7% to 71.7% while
// moving nothing correct out of it.
export function isInterpretation(term) {
  return term.standing === 'interpretive';
}

export function splitByStanding(terms) {
  const proposals = [];
  const interpretations = [];
  terms.forEach((term) => (isInterpretation(term) ? interpretations : proposals).push(term));
  return { proposals, interpretations };
}
