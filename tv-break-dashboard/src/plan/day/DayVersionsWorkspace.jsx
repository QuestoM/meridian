import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { GitCompareArrows, TriangleAlert } from 'lucide-react';
import {
  Button, Card, CardBody, EmptyState, ErrorState, InputControl, LoadingState,
  Status, TextAreaControl,
} from '../../studio';
import { Figure, Name, Prose } from '../../shell/bidi';
import { formatDay } from '../../shell/dates';
import { pageText } from '../../shell/surface-helpers';
import SideCard from './DayVersionSideCard';
import './day-versions.css';

// Competing versions of one broadcast day, read against the day as it stands.
//
// The machinery under this surface (proposals store, engine-priced rows, the
// N-way comparison with attributed reasoning) existed before the surface did;
// this file is what makes it reachable. The comparison is deliberately the
// centre of the page: the list exists to pick sides, the decision exists to
// end the argument, and everything a side claims arrives from the compare
// route with its reasoning attached rather than asserted here.

const BASE = '/api/plan/day-proposals';

async function readJson(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.reason_he || body.reason || body.detail || `${response.status}`);
    error.status = response.status;
    error.body = body;
    throw error;
  }
  return body;
}

function listProposals(day) {
  const query = day ? `?day=${encodeURIComponent(day)}` : '';
  return fetch(`${BASE}${query}`).then(readJson);
}

function compareProposals(day, proposalIds, includeLive) {
  return fetch(`${BASE}/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ day, proposal_ids: proposalIds, include_live: includeLive }),
  }).then(readJson);
}

function decideProposal(proposalId, day, verdict, note) {
  return fetch(`${BASE}/${encodeURIComponent(proposalId)}/decide`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ day, verdict, note }),
  }).then(readJson);
}

const STATUS_TONE = { proposed: 'info', adopted: 'positive', rejected: 'neutral', withdrawn: 'neutral' };

function statusLabel(status, locale) {
  const map = {
    proposed: ['Proposed', 'מוצעת'],
    adopted: ['Adopted', 'אומצה'],
    rejected: ['Rejected', 'נדחתה'],
    withdrawn: ['Withdrawn', 'נמשכה'],
  };
  const entry = map[String(status || '')] || ['Proposed', 'מוצעת'];
  return pageText(locale, entry[0], entry[1]);
}

function ProposalRow({ item, locale, checked, onToggle }) {
  const stale = Boolean(item.staleness && item.staleness.stale);
  return (
    <li>
      <label className="dvw-row" data-checked={checked ? 'true' : 'false'}>
        <InputControl
          type="checkbox"
          checked={checked}
          onChange={onToggle}
          aria-label={pageText(locale, `Include ${item.name} in the comparison`, `לכלול את ${item.name} בהשוואה`)}
        />
        <span className="dvw-row-main">
          <span className="dvw-row-name">
            <Name>{item.name}</Name>
            <Status status={STATUS_TONE[item.status] || 'neutral'}>{statusLabel(item.status, locale)}</Status>
            {stale ? (
              <Status status="warning" icon={<TriangleAlert size={12} aria-hidden="true" />}>
                {pageText(locale, 'Behind the day', 'מאחורי היום')}
              </Status>
            ) : null}
          </span>
          <span className="dvw-row-meta">
            <Name>{item.author || pageText(locale, 'Unnamed', 'ללא שם')}</Name>
            <Figure>{formatDay(item.created_at)}</Figure>
            <span>
              {Number(item.edit_count || 0) === 0
                ? pageText(locale, 'the engine day, untouched', 'יום המנוע, ללא עריכות')
                : Number(item.edit_count) === 1
                  ? pageText(locale, 'one pinned edit', 'עריכה נעוצה אחת')
                  : pageText(
                    locale,
                    `${item.edit_count} pinned edits`,
                    `${item.edit_count} עריכות נעוצות`,
                  )}
            </span>
          </span>
          {item.note ? <Prose className="dvw-row-note">{item.note}</Prose> : null}
        </span>
      </label>
    </li>
  );
}


export default function DayVersionsWorkspace({ locale = 'he', notify, refreshKey = 0 }) {
  const [day, setDay] = useState('');
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState('');
  const [selection, setSelection] = useState(() => new Set());
  const [includeLive, setIncludeLive] = useState(false);
  const [compare, setCompare] = useState(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState('');
  const [decision, setDecision] = useState(null); // {proposalId, verdict, label}
  const [note, setNote] = useState('');
  const [deciding, setDeciding] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const reload = useCallback(() => setReloadKey((key) => key + 1), []);

  useEffect(() => {
    let alive = true;
    setError('');
    setPayload(null);
    setCompare(null);
    setSelection(new Set());
    listProposals(day).then(
      (body) => { if (alive) setPayload(body); },
      (failure) => { if (alive) setError(failure.message); },
    );
    return () => { alive = false; };
  }, [day, refreshKey, reloadKey]);

  const proposals = useMemo(() => (payload && payload.proposals) || [], [payload]);

  function toggle(proposalId) {
    setSelection((current) => {
      const next = new Set(current);
      if (next.has(proposalId)) next.delete(proposalId);
      else next.add(proposalId);
      return next;
    });
  }

  function runCompare() {
    if (!payload) return;
    setComparing(true);
    setCompareError('');
    setCompare(null);
    compareProposals(payload.day, [...selection], includeLive).then(
      (body) => { setComparing(false); setCompare(body); },
      (failure) => { setComparing(false); setCompareError(failure.message); },
    );
  }

  function submitDecision() {
    if (!decision || !payload) return;
    setDeciding(true);
    decideProposal(decision.proposalId, payload.day, decision.verdict, note).then(
      () => {
        setDeciding(false);
        setDecision(null);
        setNote('');
        if (notify) {
          notify(
            decision.verdict === 'adopt' ? 'The version was adopted.' : 'The version was rejected.',
            decision.verdict === 'adopt' ? 'הגרסה אומצה.' : 'הגרסה נדחתה.',
          );
        }
        reload();
      },
      (failure) => {
        setDeciding(false);
        if (notify) notify(`The decision was refused (${failure.message}).`, `ההחלטה סורבה (${failure.message}).`);
      },
    );
  }

  const selectable = selection.size >= 2 || (selection.size >= 1 && includeLive);

  return (
    <section className="page-workspace dvw-workspace" aria-label={pageText(locale, 'Day versions', 'גרסאות היום')}>
      <header className="dvw-head">
        <div>
          <h2>{pageText(locale, 'Competing versions of the day', 'גרסאות מתחרות של היום')}</h2>
          <p>
            {pageText(
              locale,
              'Several people can put a priced version of the same day on the table. The comparison reads every side against the day as it stands, with the reasoning behind every shekel of the gap, and the decision keeps the rejected alternatives in history.',
              'כמה אנשים יכולים להניח על השולחן גרסה מתומחרת של אותו היום. ההשוואה קוראת כל צד מול היום כפי שהוא, עם הנימוק מאחורי כל שקל בפער, וההחלטה משאירה את החלופות שנדחו בהיסטוריה.',
            )}
          </p>
        </div>
        <div className="dvw-controls">
          <InputControl
            type="date"
            value={day || (payload ? payload.day : '')}
            onChange={(event) => setDay(event.target.value)}
            aria-label={pageText(locale, 'Broadcast day', 'יום שידור')}
          />
          <label className="dvw-live">
            <InputControl
              type="checkbox"
              checked={includeLive}
              onChange={(event) => setIncludeLive(event.target.checked)}
            />
            {pageText(locale, 'Include the live day', 'לכלול את היום החי')}
          </label>
          <Button type="button" onClick={runCompare} disabled={!selectable || comparing}>
            <GitCompareArrows size={16} aria-hidden="true" />
            {comparing
              ? pageText(locale, 'Comparing…', 'משווה…')
              : pageText(locale, 'Compare the selected', 'השוואת הנבחרות')}
          </Button>
        </div>
      </header>

      {error ? (
        <ErrorState
          title={pageText(locale, 'The versions could not be read', 'לא ניתן היה לקרוא את הגרסאות')}
          description={error}
          action={<Button type="button" onClick={reload}>{pageText(locale, 'Try again', 'נסו שוב')}</Button>}
        />
      ) : payload === null ? (
        <LoadingState title={pageText(locale, 'Reading the versions', 'קורא את הגרסאות')} />
      ) : payload.available === false ? (
        <EmptyState
          title={pageText(locale, 'No plan covers any day yet', 'אין עדיין תוכנית שמכסה יום כלשהו')}
          description={payload.reason_he || payload.reason || ''}
        />
      ) : (
        <>
          <p className="dvw-context">
            <Figure>{formatDay(payload.day)}</Figure>
            {' · '}
            <Name>{payload.channel}</Name>
            {' · '}
            {pageText(
              locale,
              `${proposals.length} versions on the table`,
              `${proposals.length} גרסאות על השולחן`,
            )}
            {payload.adopted ? (
              <>
                {' · '}
                {pageText(locale, 'one already adopted', 'אחת כבר אומצה')}
              </>
            ) : null}
          </p>

          {proposals.length === 0 ? (
            <EmptyState
              title={pageText(locale, 'No versions were proposed for this day', 'לא הוצעו גרסאות ליום הזה')}
              description={pageText(
                locale,
                'A version is born on the day board: pin edits onto the day and save them as a named proposal.',
                'גרסה נולדת בלוח היום: נועצים עריכות על היום ושומרים אותן כהצעה בעלת שם.',
              )}
            />
          ) : (
            <Card as="section" aria-label={pageText(locale, 'The versions', 'הגרסאות')}>
              <CardBody>
                <ul className="dvw-list">
                  {proposals.map((item) => (
                    <ProposalRow
                      key={item.proposal_id}
                      item={item}
                      locale={locale}
                      checked={selection.has(item.proposal_id)}
                      onToggle={() => toggle(item.proposal_id)}
                    />
                  ))}
                </ul>
              </CardBody>
            </Card>
          )}

          {compareError ? (
            <ErrorState
              title={pageText(locale, 'The comparison was refused', 'ההשוואה סורבה')}
              description={compareError}
            />
          ) : null}

          {compare && compare.available !== false ? (
            <section className="dvw-compare" aria-label={pageText(locale, 'The comparison', 'ההשוואה')}>
              <div className="dvw-compare-grid">
                {(compare.sides || []).map((side) => (
                  <SideCard
                    key={side.side_id}
                    side={side}
                    locale={locale}
                    highest={compare.highest_revenue_side}
                    adoptedId={payload.adopted}
                    canDecide={Boolean(side.manifest && side.manifest.proposal_id)}
                    deciding={deciding}
                    onDecide={(chosen, verdict) => {
                      setDecision({
                        proposalId: chosen.manifest.proposal_id,
                        verdict,
                        label: chosen.label,
                      });
                      setNote('');
                    }}
                  />
                ))}
              </div>
              {compare.note_he ? <p className="dvw-footnote">{compare.note_he}</p> : null}
            </section>
          ) : null}

          {decision ? (
            <Card as="section" className="dvw-decision" aria-label={pageText(locale, 'The decision', 'ההחלטה')}>
              <CardBody>
                <h4>
                  {decision.verdict === 'adopt'
                    ? pageText(locale, 'Adopting a version', 'אימוץ גרסה')
                    : pageText(locale, 'Rejecting a version', 'דחיית גרסה')}
                  {' — '}
                  <Name>{decision.label}</Name>
                </h4>
                <p>
                  {pageText(
                    locale,
                    'A decision without its reason is unreadable in a month. The note is required.',
                    'החלטה בלי הנימוק שלה אינה קריאה בעוד חודש. ההערה נדרשת.',
                  )}
                </p>
                <TextAreaControl
                  value={note}
                  onChange={(event) => setNote(event.target.value)}
                  rows={2}
                  aria-label={pageText(locale, 'Decision note', 'הערת החלטה')}
                />
                <div className="dvw-side-actions">
                  <Button type="button" onClick={submitDecision} disabled={deciding || !note.trim()}>
                    {deciding
                      ? pageText(locale, 'Recording…', 'רושם…')
                      : pageText(locale, 'Record the decision', 'רישום ההחלטה')}
                  </Button>
                  <Button type="button" variant="quiet" onClick={() => setDecision(null)} disabled={deciding}>
                    {pageText(locale, 'Cancel', 'ביטול')}
                  </Button>
                </div>
              </CardBody>
            </Card>
          ) : null}
        </>
      )}
    </section>
  );
}
