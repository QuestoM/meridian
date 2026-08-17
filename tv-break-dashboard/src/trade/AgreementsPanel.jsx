import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Card, CardBody, EmptyState, ErrorState, LoadingState, Status,
} from '../studio';
import { Button } from '../studio/actions';
import {
  ChevronLeft, ChevronRight, FilePlus2, FileText, Handshake, RefreshCcw, ShieldAlert,
  ShieldCheck,
} from 'lucide-react';
import { Code, Figure, Name } from '../shell/bidi';
import { formatSpan } from '../shell/dates';
import { formatNumber, pageText } from '../shell/format';
import {
  alarmLabel, alarmTone, ALARM_ORDER, counterpartyKind, counterpartyKindOf,
  counterpartyName, levelLabel, statusLabel, statusTone,
} from './trade-vocabulary';
import { loadAgreements, loadObligations, refusalText } from './trade-api';
import AgreementCreateFlow from './AgreementCreateFlow';
import AgreementDetailScreen from './AgreementDetailScreen';
import AgreementReviewScreen from './AgreementReviewScreen';
import './trade-agreements.css';

// Trade agreements: the record of what the channel actually promised, and the
// review that decides which of those promises the machinery is allowed to act
// on.
//
// The list is the front door and it answers one question per card: is this
// agreement acting on the business yet, and if not, what is stopping it. That is
// why the gate state sits on the card rather than inside the record. A
// commercial director looking at six agreements wants the two that are blocked.
//
// THREE STATES ARE THREE STATES. A failed read is not an empty list and an empty
// list is not a load in flight. Each one says which it is, in both languages,
// because a refusal a reviewer cannot read is a refusal aimed at nobody.

const AGREEMENT_PARAM = 'agreement';

function readParam(name) {
  if (typeof window === 'undefined') return '';
  return new URLSearchParams(window.location.search).get(name) || '';
}

// The open agreement lives in the query string so the record is addressable and
// Back returns to the list. The hash belongs to the shell's router and is left
// alone, and every other parameter on the URL is preserved.
function writeParam(name, value) {
  if (typeof window === 'undefined' || !window.history) return;
  const params = new URLSearchParams(window.location.search);
  if (value) params.set(name, value);
  else params.delete(name);
  const query = params.toString();
  window.history.replaceState(
    null, '',
    `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`,
  );
}

// The alarm summary for one approved agreement, in the order that puts a breach
// first. Counts arrive keyed by alarm from the obligations endpoint.
function alarmChips(counts, locale) {
  if (!counts) return [];
  return ALARM_ORDER
    .filter((alarm) => Number(counts[alarm]) > 0)
    .map((alarm) => ({
      alarm,
      tone: alarmTone(alarm),
      label: `${alarmLabel(alarm, locale)} · ${formatNumber(counts[alarm], locale)}`,
    }));
}

function GateLine({ row, locale }) {
  if (row.gate_blockers === null || row.gate_blockers === undefined) {
    return (
      <Status status="warning" icon={<ShieldAlert size={14} aria-hidden="true" />}>
        {pageText(locale, 'The completeness gate could not be read', 'לא ניתן היה לקרוא את שער השלמות')}
      </Status>
    );
  }
  if (row.gate_ready) {
    return (
      <Status status="positive" icon={<ShieldCheck size={14} aria-hidden="true" />}>
        {pageText(locale, 'Review complete, ready to approve', 'הסקירה הושלמה, מוכן לאישור')}
      </Status>
    );
  }
  return (
    <Status status="warning" icon={<ShieldAlert size={14} aria-hidden="true" />}>
      {pageText(
        locale,
        `${formatNumber(row.gate_blockers, locale)} blocking approval`,
        `${formatNumber(row.gate_blockers, locale)} חסמים לאישור`,
      )}
    </Status>
  );
}

function AgreementCard({ row, locale, alarms, onOpen }) {
  const Caret = locale === 'he' ? ChevronLeft : ChevronRight;
  const party = counterpartyName(row.counterparty);
  const kind = counterpartyKindOf(row.counterparty);
  const chips = alarmChips(alarms, locale);
  const opensReview = row.status === 'in_review' && Number(row.documents) > 0;
  return (
    <Button
      type="button"
      className="trd-card"
      onClick={() => onOpen(row.agreement_id, opensReview ? 'review' : 'detail')}
    >
      <span className="trd-card-head">
        <Name className="trd-card-title">{row.title}</Name>
        <Caret size={16} className="trd-card-caret" aria-hidden="true" />
      </span>

      <span className="trd-card-chips">
        <Status status={statusTone(row.status)}>{statusLabel(row.status, locale)}</Status>
        <span className="trd-chip-quiet">{levelLabel(row.level, locale)}</span>
      </span>

      <span className="trd-card-rows">
        {party ? (
          <span className="trd-card-row">
            <span className="trd-card-label">{counterpartyKind(kind, locale)}</span>
            <Name className="trd-card-value">{party}</Name>
          </span>
        ) : null}
        {row.window ? (
          <span className="trd-card-row">
            <span className="trd-card-label">{pageText(locale, 'Effective window', 'תקופת תוקף')}</span>
            <Figure className="trd-card-value">{formatSpan(row.window.from, row.window.to, locale)}</Figure>
          </span>
        ) : null}
        <span className="trd-card-row">
          <span className="trd-card-label">{pageText(locale, 'Documents', 'מסמכים')}</span>
          <Figure className="trd-card-value">{formatNumber(row.documents, locale)}</Figure>
        </span>
      </span>

      <span className="trd-card-gate"><GateLine row={row} locale={locale} /></span>

      {chips.length > 0 ? (
        <span className="trd-card-alarms">
          <span className="trd-card-label">{pageText(locale, 'Commitment standing', 'מצב התחייבויות')}</span>
          <span className="trd-alarm-chips">
            {chips.map((chip) => (
              <Status key={chip.alarm} status={chip.tone}>{chip.label}</Status>
            ))}
          </span>
        </span>
      ) : null}

      <span className="trd-card-open">
        {opensReview
          ? pageText(locale, 'Open the review', 'פתיחת הסקירה')
          : pageText(locale, 'Open the agreement', 'פתיחת ההסכם')}
      </span>
      <Code className="trd-card-id">{row.agreement_id}</Code>
    </Button>
  );
}

export default function AgreementsPanel({ locale = 'he', notify = () => {}, canEdit = true, editRefusal = '', refreshKey = 0 }) {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);
  // Keyed by agreement id: the alarm counts of an approved agreement. Absent
  // means not read; an agreement with no approved version is never read at all
  // because it cannot have a standing.
  const [alarms, setAlarms] = useState({});
  const [creating, setCreating] = useState(false);
  const [open, setOpen] = useState(() => readParam(AGREEMENT_PARAM));
  const [mode, setMode] = useState('detail');
  const [reloadKey, setReloadKey] = useState(0);
  const reload = useCallback(() => setReloadKey((key) => key + 1), []);

  useEffect(() => {
    let alive = true;
    setError(null);
    setRows(null);
    loadAgreements().then(
      (payload) => { if (alive) setRows(payload.agreements || []); },
      (failure) => { if (alive) { setError(failure); setRows([]); } },
    );
    return () => { alive = false; };
  }, [refreshKey, reloadKey]);

  // Standings are read only for agreements that have an approved version, and
  // one failure never removes the card: a card whose standing could not be read
  // simply shows no standing rather than an invented healthy one.
  useEffect(() => {
    if (!rows || rows.length === 0) return undefined;
    let alive = true;
    const approved = rows.filter((row) => row.current_version_id);
    Promise.allSettled(approved.map((row) => loadObligations(row.agreement_id))).then((results) => {
      if (!alive) return;
      const next = {};
      results.forEach((result, index) => {
        if (result.status !== 'fulfilled' || !result.value.available) return;
        next[approved[index].agreement_id] = result.value.alarm_counts || {};
      });
      setAlarms(next);
    });
    return () => { alive = false; };
  }, [rows]);

  useEffect(() => { writeParam(AGREEMENT_PARAM, open); }, [open]);

  const openRow = useMemo(
    () => (rows || []).find((row) => row.agreement_id === open) || null,
    [rows, open],
  );

  function openAgreement(agreementId, nextMode) {
    setMode(nextMode);
    setOpen(agreementId);
  }

  function closeAgreement() {
    setOpen('');
    reload();
  }

  // A record whose id is in the URL but not in the list is a fact worth stating:
  // it means the agreement was removed, or the address is wrong. The list is
  // still shown behind the message.
  if (open && rows && !openRow) {
    return (
      <ErrorState
        title={pageText(locale, 'That agreement is not on file', 'ההסכם הזה אינו קיים')}
        description={pageText(
          locale,
          'The address names an agreement the store does not hold. It may have been withdrawn, or the link may be stale.',
          'הכתובת מפנה להסכם שאינו נמצא במאגר. ייתכן שהוא בוטל, או שהקישור אינו עדכני.',
        )}
        action={(
          <Button type="button" onClick={closeAgreement}>
            {pageText(locale, 'Back to the agreements', 'חזרה לרשימת ההסכמים')}
          </Button>
        )}
      />
    );
  }

  if (openRow && mode === 'review') {
    return (
      <AgreementReviewScreen
        agreementId={openRow.agreement_id}
        locale={locale}
        notify={notify}
        canEdit={canEdit}
        editRefusal={editRefusal}
        onClose={closeAgreement}
        onOpenDetail={() => setMode('detail')}
      />
    );
  }

  if (openRow) {
    return (
      <AgreementDetailScreen
        agreementId={openRow.agreement_id}
        locale={locale}
        notify={notify}
        canEdit={canEdit}
        editRefusal={editRefusal}
        onClose={closeAgreement}
        onOpenReview={() => setMode('review')}
      />
    );
  }

  return (
    <section className="trd-panel" aria-busy={rows === null}>
      <header className="page-header">
        <div>
          <h2>{pageText(locale, 'Trade agreements', 'הסכמי סחר')}</h2>
          <p>
            {pageText(
              locale,
              'Every framework, advertiser agreement and amendment the channel has signed, with what each one will do to pricing, placement and settlement once it is approved.',
              'כל הסכם מסגרת, הסכם מפרסם ותיקון שהערוץ חתם עליהם, ולצידם מה כל אחד מהם יעשה לתמחור, לשיבוץ ולהתחשבנות ברגע שיאושר.',
            )}
          </p>
        </div>
        <div className="trd-header-actions">
          <Button type="button" variant="outlined" onClick={reload}>
            <RefreshCcw size={14} aria-hidden="true" />
            {pageText(locale, 'Refresh', 'רענון')}
          </Button>
          {canEdit ? (
            <Button type="button" onClick={() => setCreating(true)}>
              <FilePlus2 size={14} aria-hidden="true" />
              {pageText(locale, 'Add an agreement', 'הוספת הסכם')}
            </Button>
          ) : null}
        </div>
      </header>

      {!canEdit && editRefusal ? (
        <Card dense className="trd-refusal">
          <CardBody><p>{editRefusal}</p></CardBody>
        </Card>
      ) : null}

      {rows === null ? (
        <LoadingState
          title={pageText(locale, 'Reading the agreements', 'קורא את ההסכמים')}
          description={pageText(
            locale,
            'The store is being read. Nothing is shown until it answers.',
            'המאגר בקריאה. לא מוצג דבר עד שיענה.',
          )}
        />
      ) : null}

      {error ? (
        <ErrorState
          title={pageText(locale, 'The agreements could not be read', 'לא ניתן היה לקרוא את ההסכמים')}
          description={refusalText(error, locale)}
          action={(
            <Button type="button" onClick={reload}>
              {pageText(locale, 'Try again', 'נסו שוב')}
            </Button>
          )}
        />
      ) : null}

      {rows && !error && rows.length === 0 ? (
        <EmptyState
          title={pageText(locale, 'No agreements on file yet', 'אין עדיין הסכמים במאגר')}
          description={pageText(
            locale,
            'An agreement enters here as a signed document. It is read, every clause is reviewed against the term catalogue, and only what a person approves is allowed to change pricing or placement.',
            'הסכם נכנס לכאן כמסמך חתום. הוא נקרא, כל סעיף נסקר מול קטלוג המונחים, ורק מה שאדם מאשר רשאי לשנות תמחור או שיבוץ.',
          )}
          action={canEdit ? (
            <Button type="button" onClick={() => setCreating(true)}>
              <FilePlus2 size={14} aria-hidden="true" />
              {pageText(locale, 'Add the first agreement', 'הוספת ההסכם הראשון')}
            </Button>
          ) : null}
        />
      ) : null}

      {rows && rows.length > 0 ? (
        <>
          <p className="trd-count" role="status">
            <Handshake size={14} aria-hidden="true" />
            {pageText(
              locale,
              `${formatNumber(rows.length, locale)} agreements on file`,
              `${formatNumber(rows.length, locale)} הסכמים במאגר`,
            )}
          </p>
          <div className="trd-grid">
            {rows.map((row) => (
              <AgreementCard
                key={row.agreement_id}
                row={row}
                locale={locale}
                alarms={alarms[row.agreement_id]}
                onOpen={openAgreement}
              />
            ))}
          </div>
        </>
      ) : null}

      <Card dense className="trd-boundary">
        <CardBody>
          <FileText size={16} aria-hidden="true" />
          <p>
            {pageText(
              locale,
              'An approved agreement writes rules into the live pricing, frequency and settlement stores. A term the engine cannot bind is named as such before approval, never after, and it changes nothing.',
              'הסכם מאושר כותב כללים למאגרי התמחור, התדירות וההתחשבנות הפעילים. מונח שהמנוע אינו יכול לחבר מסומן ככזה לפני האישור, לא אחריו, והוא אינו משנה דבר.',
            )}
          </p>
        </CardBody>
      </Card>

      {creating ? (
        <AgreementCreateFlow
          locale={locale}
          notify={notify}
          onClose={() => setCreating(false)}
          onCreated={(agreementId, nextMode) => {
            setCreating(false);
            reload();
            if (agreementId) openAgreement(agreementId, nextMode || 'detail');
          }}
        />
      ) : null}
    </section>
  );
}
