import React, { useCallback, useEffect, useState } from 'react';
import { Figure } from '../../shell/bidi';
import { WALLS, fetchSession, payloadCanEdit } from '../../session';
import MakeGoodLedger from './MakeGoodLedger';
import PacingBoard from './PacingBoard';
import { acceptRisk, loadBoard, loadLedger, moveMakeGood, raiseMakeGood, refusalText } from './pacing-api';
import { isolate, pick, term, vocabularyLabel } from './pacing-helpers';
import './pacing.css';
import './pacing-row.css';
import './pacing-days.css';
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
  // The refusal a write came back with, held here and printed on this surface.
  // notify() is the product's own channel for this and it is a no-op at the
  // address this panel is mounted at: measured, workspace-router.jsx renders the
  // Campaigns destination without a notify prop, so every notice this panel sends
  // is swallowed and a refused write said nothing at all for 2.5 s of polling. A
  // refusal the server took the trouble to word in two languages has to reach the
  // person who was refused, whatever the shell around it does.
  const [refusal, setRefusal] = useState('');
  const [focusCampaign, setFocusCampaign] = useState('');

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

  // A campaign named on a ledger record opens that campaign's own row. Without
  // this the name went back to the board and left the reader on whichever row was
  // first, which on this data is a different campaign from the one they clicked.
  const clearFocus = useCallback(() => setFocusCampaign(''), []);
  const openCampaign = useCallback((id) => {
    if (onOpenCampaign) {
      onOpenCampaign(id);
      return;
    }
    setFocusCampaign(id);
    setView(BOARD);
  }, [onOpenCampaign]);

  useEffect(() => {
    let active = true;
    fetchSession().then((record) => { if (active) setSession(record); });
    return () => { active = false; };
  }, []);

  // The gate is a pair, the canEdit answer and the refusal that goes with it,
  // which is the shape every other destination reads. Held as one object it is
  // always truthy, so a read-only account was shown a control the server would
  // then refuse, and the refusal it was shown came from a constant here rather
  // than from the wall that made the decision.
  const gate = payloadCanEdit(board.payload, session, WALLS.readOnlyRole);
  const canEdit = gate.canEdit;
  const editRefusal = gate.reason || WALLS.readOnlyRole.detail;

  // One place that words a refusal, so the three writes cannot state one three
  // ways. A server that answered with no detail at all still leaves the reader a
  // sentence, because a failure that says nothing is the worst of the three.
  function refuse(error, en, he) {
    const said = refusalText(error, locale === 'he' ? 'he' : 'en');
    const opener = pick(locale, en, he);
    setRefusal(said ? `${opener} ${said}` : opener);
    notify(`${en} ${refusalText(error, 'en')}`, `${he} ⁦${refusalText(error, 'he')}⁩`);
  }

  async function onRaise(row) {
    setBusyId(row.campaign_id);
    try {
      const answer = await raiseMakeGood(row.campaign_id, '');
      const record = answer.make_good;
      setRefusal('');
      notify(
        `Make-good ${record.make_good_id} raised for ${row.name}.`,
        `פיצוי שידור ⁦${record.make_good_id}⁩ נפתח עבור ⁦${row.name}⁩.`,
      );
      reload();
      setView(LEDGER);
    } catch (error) {
      refuse(
        error,
        'The make-good could not be raised.',
        'לא ניתן היה לפתוח את פיצוי השידור.',
      );
    } finally {
      setBusyId('');
    }
  }

  // The other ending. It writes a record and changes no figure, so the notice
  // says what was recorded rather than claiming the campaign moved.
  async function onAccept(row) {
    setBusyId(row.campaign_id);
    try {
      const answer = await acceptRisk(row.campaign_id, '');
      const record = answer.make_good;
      setRefusal('');
      notify(
        `The risk on ${row.name} is recorded as taken on, ${record.make_good_id}.`,
        `הסיכון ב⁦${row.name}⁩ נרשם כמתקבל, ⁦${record.make_good_id}⁩.`,
      );
      reload();
    } catch (error) {
      refuse(
        error,
        'The risk could not be recorded.',
        'לא ניתן היה לרשום את קבלת הסיכון.',
      );
    } finally {
      setBusyId('');
    }
  }

  // Answers whether the move landed, so the form that raised it can keep what the
  // reader typed when it did not. It used to close the offer form before the
  // request had answered, which threw away a value, a window and a note on every
  // refusal and made the reader type all three again.
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
      setRefusal('');
      notify(
        `Make-good ${makeGoodId} is now ${vocabularyLabel(states, landed, 'en')}.`,
        `פיצוי שידור ⁦${makeGoodId}⁩ נמצא כעת במצב ⁦${vocabularyLabel(states, landed, 'he')}⁩.`,
      );
      reload();
      return true;
    } catch (error) {
      refuse(
        error,
        'The make-good could not be moved.',
        'לא ניתן היה להעביר את פיצוי השידור.',
      );
      return false;
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
    const settled = decided();
    return pick(
      locale,
      `${acting - settled} of ${counts.total || 0} campaigns still need a decision, ${settled} of the ${acting} at risk already carry one, ${counts.unknown || 0} cannot be paced yet.`,
      `${isolate(acting - settled)} מתוך ${isolate(counts.total || 0)} קמפיינים עדיין דורשים החלטה, ל־${isolate(settled)} מתוך ${isolate(acting)} שבסיכון כבר יש אחת, ${isolate(counts.unknown || 0)} עדיין לא ניתנים למדידת קצב.`,
    );
  }

  // How many of the rows the board is asking a decision about already carry one.
  // The job this destination serves is done when every at-risk campaign has an
  // act taken against it or a recorded decision to take the risk on, so the
  // headline counts what is left rather than what exists.
  function decided() {
    if (board.status !== 'ready') return 0;
    const payload = board.payload;
    const asking = payload.needs_a_decision || [];
    const raised = payload.make_goods || {};
    const accepted = payload.acceptances || {};
    return (payload.rows || []).filter((row) => (
      asking.indexOf(row.headline.verdict) >= 0
      && ((raised[row.campaign_id] || []).length > 0 || (accepted[row.campaign_id] || []).length > 0)
    )).length;
  }

  // How many of the counted rows the demo seed wrote. A count that mixes seeded
  // rows into an operational one reads as a morning's work, and on this data
  // most of them are seeded, so the count says which is which. The rows
  // themselves are marked; the sentence above them was not.
  function seeded() {
    if (board.status !== 'ready') return null;
    const counts = board.payload.counts || {};
    if (!counts.demo) return null;
    return pick(
      locale,
      `${counts.demo} of the ${counts.total || 0} are demo rows the seed wrote against the real traffic log, not campaigns an operator booked. Their goals and flight dates are the seed's.`,
      `${isolate(counts.demo)} מתוך ${isolate(counts.total || 0)} הן שורות הדגמה שנכתבו על בסיס יומן השידור האמיתי ולא קמפיינים שמפעיל הזמין. היעדים ותאריכי הטיסה שלהן הם של זרע ההדגמה.`,
    );
  }

  return (
    <section className="page-workspace pacing-workspace">
      {/* The heading is an h2 and the two view controls ride the same row as it.
          Mounted inside a destination that already carries an h1 and a view
          strip, a second h1 above a second strip is two documents on one page,
          and it was measured costing 90 px of the fold before the first row.
          Measured before: the first row sat at y=531 in an 851 px viewport. */}
      <div className="page-header pacing-header">
        <div>
          {/* The two words this destination is about come from the product
              vocabulary, not from here. It had drifted to קצב where the
              controlled word is קצב אספקה. */}
          <h2>
            {pick(
              locale,
              `${term('concept.pacing', 'en')} and make-good`,
              `${term('concept.pacing', 'he')} ו${term('object.make_good', 'he')}`,
            )}
          </h2>
          <p>
            {headline()}
            {seeded() ? ' ' : ''}
            {seeded() ? <span className="pacing-seeded">{seeded()}</span> : null}
          </p>
        </div>
        <nav className="pacing-views" role="tablist" aria-label={pick(locale, 'Pacing views', 'תצוגות קצב')}>
          <button type="button" role="tab" aria-selected={view === BOARD}
                  className={view === BOARD ? 'active' : ''} onClick={() => setView(BOARD)}>
            {pick(locale, 'Campaign pacing', 'קצב אספקה של הקמפיינים')}
          </button>
          <button type="button" role="tab" aria-selected={view === LEDGER}
                  className={view === LEDGER ? 'active' : ''} onClick={() => setView(LEDGER)}>
            {pick(locale, 'Decision ledger', 'ספר ההחלטות')}
            {ledger.status === 'ready' && (ledger.payload.open_count + ledger.payload.accepted_count)
              ? (
                <Figure className="pacing-open-count">
                  {ledger.payload.open_count + ledger.payload.accepted_count}
                </span>
              )
              : null}
          </button>
          <button type="button" className="pacing-refresh" onClick={reload}>
            {pick(locale, 'Read again', 'קראו שוב')}
          </button>
        </nav>
      </div>

      {refusal ? (
        <div className="pacing-refusal" role="alert">
          <p>{refusal}</p>
          <button type="button" onClick={() => setRefusal('')}>{pick(locale, 'Dismiss', 'סגרו')}</button>
        </div>
      ) : null}

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
          onAccept={onAccept}
          onOpenMakeGood={() => setView(LEDGER)}
          focusCampaignId={focusCampaign}
          onFocused={clearFocus}
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
          onOpenCampaign={openCampaign}
        />
      ) : null}
      {view === LEDGER && ledger.status === 'failed' ? (
        <div className="pacing-failed" role="alert">
          <p>
            {pick(
              locale,
              'The decision ledger could not be read. What is missing is a failure, not an empty result.',
              'לא ניתן היה לקרוא את ספר ההחלטות. מה שחסר הוא כשל, לא תוצאה ריקה.',
            )}
          </p>
          <button type="button" onClick={reload}>{pick(locale, 'Try again', 'נסו שוב')}</button>
        </div>
      ) : null}
      {view === LEDGER && ledger.status === 'loading' ? (
        <p className="pacing-loading">{pick(locale, 'Reading the decision ledger', 'קורא את ספר ההחלטות')}</p>
      ) : null}
    </section>
  );
}
