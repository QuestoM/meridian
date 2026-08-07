import React, { useCallback, useEffect, useState } from 'react';
import { WALLS, fetchSession, payloadCanEdit } from '../../session';
import MakeGoodLedger from './MakeGoodLedger';
import PacingBoard from './PacingBoard';
import { loadBoard, loadLedger, moveMakeGood, raiseMakeGood, refusalText } from './pacing-api';
import { isolate, pick, vocabularyLabel } from './pacing-helpers';
import './pacing.css';
import './makegood.css';

// Clients, pacing: whether a campaign is behind before it is too late to fix, and
// the ledger of what is owed when it is.
//
// The two views are one destination because they are one job. A shortfall is
// measured on the board and it is settled in the ledger, and an account manager
// who had to leave one to reach the other would carry the figure in their head.
//
// The read has four states and never three. A read in flight, a read that landed,
// a read that failed, and a landed read with nothing in it are four different
// facts, and collapsing the last two is how a screen comes to print a confident
// zero over a request that never answered.

const BOARD = 'board';
const LEDGER = 'ledger';

export default function PacingWorkspace({ locale = 'he', notify = () => {}, refreshKey = 0, onOpenCampaign }) {
  const he = locale === 'he';
  const [view, setView] = useState(BOARD);
  const [board, setBoard] = useState({ status: 'loading', payload: null });
  const [ledger, setLedger] = useState({ status: 'loading', payload: null });
  const [session, setSession] = useState(null);
  const [busyId, setBusyId] = useState('');

  const reload = useCallback(() => {
    let active = true;
    setBoard((current) => ({ status: current.payload ? 'ready' : 'loading', payload: current.payload }));
    loadBoard()
      .then((payload) => { if (active) setBoard({ status: 'ready', payload }); })
      .catch(() => { if (active) setBoard({ status: 'failed', payload: null }); });
    loadLedger()
      .then((payload) => { if (active) setLedger({ status: 'ready', payload }); })
      .catch(() => { if (active) setLedger({ status: 'failed', payload: null }); });
    return () => { active = false; };
  }, []);

  useEffect(() => reload(), [reload, refreshKey]);
  useEffect(() => {
    let active = true;
    fetchSession().then((record) => { if (active) setSession(record); });
    return () => { active = false; };
  }, []);

  const canEdit = payloadCanEdit(board.payload, session, WALLS.readOnlyRole);
  const editRefusal = WALLS.readOnlyRole.detail;

  async function onRaise(row) {
    setBusyId(row.campaign_id);
    try {
      const answer = await raiseMakeGood(row.campaign_id, '');
      const record = answer.make_good;
      notify(
        `Make-good ${record.make_good_id} raised for ${row.name}.`,
        `פיצוי שידור ⁦${record.make_good_id}⁩ נפתח עבור ⁦${row.name}⁩.`,
      );
      reload();
      setView(LEDGER);
    } catch (error) {
      notify(
        `The make-good could not be raised. ${refusalText(error, 'en')}`,
        `לא ניתן היה לפתוח את פיצוי השידור. ⁦${refusalText(error, 'he')}⁩`,
      );
    } finally {
      setBusyId('');
    }
  }

  async function onMove(makeGoodId, payload) {
    setBusyId(makeGoodId);
    try {
      const answer = await moveMakeGood(makeGoodId, payload);
      // The state key is the store's word, not a reader's. The notice says the
      // state in the language of the person who pressed the control, from the
      // vocabulary the ledger publishes, so no internal key reaches a screen.
      const states = (ledger.payload && ledger.payload.vocabulary
        ? ledger.payload.vocabulary.states
        : []);
      const landed = answer.make_good.state;
      notify(
        `Make-good ${makeGoodId} is now ${vocabularyLabel(states, landed, 'en')}.`,
        `פיצוי שידור ⁦${makeGoodId}⁩ נמצא כעת במצב ⁦${vocabularyLabel(states, landed, 'he')}⁩.`,
      );
      reload();
    } catch (error) {
      notify(
        `The make-good could not be moved. ${refusalText(error, 'en')}`,
        `לא ניתן היה להעביר את פיצוי השידור. ⁦${refusalText(error, 'he')}⁩`,
      );
    } finally {
      setBusyId('');
    }
  }

  function headline() {
    if (board.status === 'loading') {
      return pick(locale, 'Reading the pacing board', 'קורא את לוח הקצב');
    }
    if (board.status === 'failed') {
      return pick(
        locale,
        'The pacing board could not be read, so no count is shown rather than a zero.',
        'לא ניתן היה לקרוא את לוח הקצב, ולכן לא מוצג מספר במקום אפס.',
      );
    }
    const counts = board.payload.counts || {};
    const acting = (counts.behind || 0) + (counts.at_risk || 0);
    return pick(
      locale,
      `${acting} of ${counts.total || 0} campaigns need a decision today, ${counts.unknown || 0} cannot be paced yet.`,
      `${isolate(acting)} מתוך ${isolate(counts.total || 0)} קמפיינים דורשים החלטה היום, ${isolate(counts.unknown || 0)} עדיין לא ניתנים למדידת קצב.`,
    );
  }

  return (
    <section className="page-workspace pacing-workspace" dir={he ? 'rtl' : 'ltr'}>
      <div className="page-header">
        <div>
          <h1>{pick(locale, 'Pacing and make-good', 'קצב ופיצוי שידור')}</h1>
          <p>{headline()}</p>
        </div>
        <button type="button" className="pacing-refresh" onClick={reload}>
          {pick(locale, 'Read again', 'קראו שוב')}
        </button>
      </div>

      <nav className="pacing-views" role="tablist" aria-label={pick(locale, 'Pacing views', 'תצוגות קצב')}>
        <button type="button" role="tab" aria-selected={view === BOARD}
                className={view === BOARD ? 'active' : ''} onClick={() => setView(BOARD)}>
          {pick(locale, 'Campaign pacing', 'קצב הקמפיינים')}
        </button>
        <button type="button" role="tab" aria-selected={view === LEDGER}
                className={view === LEDGER ? 'active' : ''} onClick={() => setView(LEDGER)}>
          {pick(locale, 'Make-good ledger', 'ספר פיצויי השידור')}
          {ledger.status === 'ready' && ledger.payload.open_count
            ? <span className="pacing-open-count" dir="ltr">{ledger.payload.open_count}</span>
            : null}
        </button>
      </nav>

      {view === BOARD && board.status === 'loading' ? (
        <p className="pacing-loading">{pick(locale, 'Reading the pacing board', 'קורא את לוח הקצב')}</p>
      ) : null}
      {view === BOARD && board.status === 'failed' ? (
        <div className="pacing-failed" role="alert">
          <p>
            {pick(
              locale,
              'The pacing board could not be read. What is missing is a failure, not an empty result.',
              'לא ניתן היה לקרוא את לוח הקצב. מה שחסר הוא כשל, לא תוצאה ריקה.',
            )}
          </p>
          <button type="button" onClick={reload}>{pick(locale, 'Try again', 'נסו שוב')}</button>
        </div>
      ) : null}
      {view === BOARD && board.status === 'ready' ? (
        <PacingBoard
          payload={board.payload}
          locale={locale}
          canEdit={canEdit}
          editRefusal={editRefusal}
          busyId={busyId}
          onRaise={onRaise}
          onOpenMakeGood={() => setView(LEDGER)}
        />
      ) : null}

      {view === LEDGER && ledger.status === 'ready' ? (
        <MakeGoodLedger
          payload={ledger.payload}
          locale={locale}
          canEdit={canEdit}
          editRefusal={editRefusal}
          busyId={busyId}
          onMove={onMove}
          onOpenCampaign={(id) => (onOpenCampaign ? onOpenCampaign(id) : setView(BOARD))}
        />
      ) : null}
      {view === LEDGER && ledger.status === 'failed' ? (
        <div className="pacing-failed" role="alert">
          <p>
            {pick(
              locale,
              'The make-good ledger could not be read. What is missing is a failure, not an empty result.',
              'לא ניתן היה לקרוא את ספר פיצויי השידור. מה שחסר הוא כשל, לא תוצאה ריקה.',
            )}
          </p>
          <button type="button" onClick={reload}>{pick(locale, 'Try again', 'נסו שוב')}</button>
        </div>
      ) : null}
      {view === LEDGER && ledger.status === 'loading' ? (
        <p className="pacing-loading">{pick(locale, 'Reading the make-good ledger', 'קורא את ספר פיצויי השידור')}</p>
      ) : null}
    </section>
  );
}
