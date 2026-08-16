import React, { useCallback, useEffect, useState } from 'react';
import { Button } from '../../studio/actions';
import { Figure } from '../../shell/bidi';
import { WALLS, fetchSession, payloadCanEdit } from '../../session';
import MakeGoodLedger from './MakeGoodLedger';
import PacingBoard from './PacingBoard';
import { acceptRisk, loadBoard, loadLedger, moveMakeGood, raiseMakeGood, refusalOpens, refusalText } from './pacing-api';
import { isolate, localized, pick, term, vocabularyLabel } from './pacing-helpers';
import { rememberCampaign, takeRememberedCampaign } from './pacing-place';
import { headlineSentence, seededSentence } from './pacing-summary';
import './pacing.css';
import './pacing-row.css';
import './pacing-row-collapsed.css';
import './pacing-days.css';
import './makegood.css';

// One destination for measuring a shortfall and settling what it creates.

const BOARD = 'board';
const LEDGER = 'ledger';

// A read that failed, with the act that retries it, for both views.
function Failed({ locale, en, he, onRetry }) {
  return (
    <div className="pacing-failed" role="alert">
      <p>{pick(locale, en, he)}</p>
      <Button type="button" onClick={onRetry}>{pick(locale, 'Try again', 'נסו שוב')}</Button>
    </div>
  );
}

export default function PacingWorkspace({ locale = 'he', notify = () => {}, refreshKey = 0, onOpenCampaign }) {
  const [view, setView] = useState(BOARD);
  const [board, setBoard] = useState({ status: 'loading', payload: null });
  const [ledger, setLedger] = useState({ status: 'loading', payload: null });
  const [session, setSession] = useState(null);
  const [busyId, setBusyId] = useState('');
  // Keep write outcomes local even when the outer notification channel is absent.
  const [refusal, setRefusal] = useState('');
  const [notice, setNotice] = useState('');
  // A refusal can name the campaign or decision that should open instead.
  const [refusalOpen, setRefusalOpen] = useState(null);
  const [focusCampaign, setFocusCampaign] = useState('');
  // Which record a control asked for, so the ledger opens on it rather than at
  // the top of itself. Both seams that name one passed an id into a handler that
  // threw it away.
  const [focusMakeGood, setFocusMakeGood] = useState('');
  // The record the last write created, and the published transition that reverses
  // it. Measured: pressing a on a focused row recorded MG_0001 immediately and
  // the banner confirming it offered Dismiss and nothing else.
  const [undoable, setUndoable] = useState(null);

  // A refusal is worded in the language it was raised in and then held in state,
  // so a reader who switches language is left with the other one. Measured: an
  // English refusal sat in a Hebrew panel until the next write. It goes when the
  // language does, because a stale sentence is worse than none.
  useEffect(() => { setRefusal(''); setRefusalOpen(null); setNotice(''); setUndoable(null); }, [locale]);

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
      // The way out of this board is also the way back to it. Leaving by a name
      // unmounts this panel, so the row a reader left from is written down and
      // the next mount reads it once. Measured before this: the return trip cost
      // a click on the Pacing tab and a rescroll through 56 rows.
      rememberCampaign(id);
      onOpenCampaign(id);
      return;
    }
    setFocusCampaign(id);
    setView(BOARD);
  }, [onOpenCampaign]);

  // A control that names a record opens that record. Both seams called this with
  // a make-good id and the panel answered with a plain switch to the ledger,
  // dropping it, so "Risk taken on, open the record" landed on an unscrolled,
  // unmarked list. It is the mirror of the ledger name that focuses a board row.
  const openMakeGood = useCallback((makeGoodId) => {
    setFocusMakeGood(String(makeGoodId || ''));
    setView(LEDGER);
  }, []);
  const clearMakeGoodFocus = useCallback(() => setFocusMakeGood(''), []);

  // The row a reader left this panel by, focused once when they come back.
  useEffect(() => {
    if (board.status !== 'ready') return;
    const remembered = takeRememberedCampaign();
    if (remembered) setFocusCampaign(remembered);
  }, [board.status]);

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
  // The refusal in the language the reader is in. The wall that decides it holds
  // one Hebrew constant and is a frozen wave-zero module, so an English reader on
  // a viewer account met לחשבון צפייה אין הרשאת עריכה with every other word on
  // the screen in English. This piece's own reads publish that same refusal as a
  // pair, with the Hebrew taken from the wall's constant rather than copied, and
  // the single-language string stays the fallback for anything the pair does not
  // cover.
  const refusalPair = localized(board.payload, 'can_edit_reason', locale)
    || localized(ledger.payload, 'can_edit_reason', locale);
  const editRefusal = refusalPair || gate.reason || WALLS.readOnlyRole.detail;

  // One place that words a refusal, so the three writes cannot state one three
  // ways. A server that answered with no detail at all still leaves the reader a
  // sentence, because a failure that says nothing is the worst of the three.
  function refuse(error, en, he) {
    const raw = refusalText(error, locale === 'he' ? 'he' : 'en');
    // The wall answers a refused write with the same single-language constant it
    // stamped on the read, so the sentence a reader met before the click and the
    // one they meet after it are worded here the same way.
    const said = raw && raw === gate.reason && refusalPair ? refusalPair : raw;
    const opener = pick(locale, en, he);
    setRefusal(said ? `${opener} ${said}` : opener);
    setRefusalOpen(refusalOpens(error));
    setNotice('');
    setUndoable(null);
    notify(`${en} ${refusalText(error, 'en')}`, `${he} ${isolate(refusalText(error, 'he'))}`);
  }

  // A write that landed says so here as well as through the shell's channel, and
  // it clears whatever the last refusal was, because the two cannot both be true
  // about the same act. A write that can be reversed hands the record and the
  // published transition that reverses it, and the banner carries the control.
  function announce(en, he, reversible) {
    clearRefusal();
    setNotice(pick(locale, en, he));
    setUndoable(reversible || null);
    notify(en, he);
  }

  // Reversing the decision that was just written. The transition and the reason
  // are the ledger's own, published on the answer to the act, so this surface
  // holds no second copy of what an undo is. It goes through the same move as
  // every other transition, so it is refused the same way and lands as a
  // withdrawal with an actor and an instant rather than deleting anything.
  function onUndo(pending) {
    setNotice('');
    setUndoable(null);
    return onMove(pending.id, { state: pending.undo.state, reason: pending.undo.reason, note: '' });
  }

  // The refusal named somewhere to go, so the banner takes the reader there.
  function clearRefusal() {
    setRefusal('');
    setRefusalOpen(null);
  }

  function followRefusal() {
    const opens = refusalOpen;
    clearRefusal();
    if (!opens) return;
    if (opens.kind === 'campaign') {
      openCampaign(opens.id);
      return;
    }
    openMakeGood(opens.id);
  }

  async function onRaise(row) {
    setBusyId(row.campaign_id);
    try {
      const answer = await raiseMakeGood(row.campaign_id, '');
      const record = answer.make_good;
      // The name is isolated in both languages and not only in Hebrew. A campaign
      // name here is a period, then the advertiser, then the brand, and two of
      // the three are Hebrew: dropped bare into an English sentence the neutral
      // separator between the two Hebrew runs takes their direction and the
      // screen names the brand before the advertiser.
      //
      // Every isolate here is isolate(), the FIRST-STRONG pair. These four
      // sentences held a hand-typed LEFT-TO-RIGHT one, which lays a Hebrew run
      // out left to right: measured, the Hebrew acceptance notice wrapped the
      // campaign name in U+2066. A Hebrew phrase in a Hebrew sentence takes no
      // isolate at all, so the move notice's two vocabulary words lost theirs.
      announce(
        `Make-good ${record.make_good_id} raised for ${isolate(row.name)}.`,
        `פיצוי שידור ${isolate(record.make_good_id)} נפתח עבור ${isolate(row.name)}.`,
        answer.undo ? { id: record.make_good_id, undo: answer.undo } : null,
      );
      reload();
      openMakeGood(record.make_good_id);
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
      announce(
        `The risk on ${isolate(row.name)} is recorded as taken on, ${record.make_good_id}.`,
        `הסיכון ב${isolate(row.name)} נרשם כמתקבל, ${isolate(record.make_good_id)}.`,
        answer.undo ? { id: record.make_good_id, undo: answer.undo } : null,
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
      // Neither the state nor the noun is a store key on a screen; both are the
      // ledger's published words, the same ones the row is labelled from.
      // Measured in a browser: revoking a recorded risk acceptance announced
      // "Make-good MG_0001 is now Withdrawn" about a row whose own chip read
      // Risk acceptance. The Hebrew states the record's state rather than saying
      // the record is in it, because one kind is masculine and one feminine.
      const words = (ledger.payload && ledger.payload.vocabulary) || {};
      const { state: landed, kind: noun } = answer.make_good;
      announce(
        `${vocabularyLabel(words.kinds, noun, 'en')} ${makeGoodId} is now ${vocabularyLabel(words.states, landed, 'en')}.`,
        `המצב של ${vocabularyLabel(words.kinds, noun, 'he')} ${isolate(makeGoodId)} הוא כעת ${vocabularyLabel(words.states, landed, 'he')}.`,
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

  // The two counting sentences are prose about a payload and hold no JSX, so they
  // live in pacing-summary.js where node can execute them against the shipped
  // board rather than a guard having to grep this file for a substring of one.
  const seededLine = seededSentence(board, locale);

  function moveView(event) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const order = [BOARD, LEDGER];
    const current = order.indexOf(view);
    let next = current;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = order.length - 1;
    if (event.key === 'ArrowRight') next = (current + (locale === 'he' ? -1 : 1) + order.length) % order.length;
    if (event.key === 'ArrowLeft') next = (current + (locale === 'he' ? 1 : -1) + order.length) % order.length;
    setView(order[next]);
    event.currentTarget.querySelector(`[data-pacing-view="${order[next]}"]`)?.focus();
  }

  return (
    <section className="page-workspace pacing-workspace">
      {/* The heading is an h2 and the two view controls ride the same row as it.
          Mounted inside a destination that already carries an h1 and a view
          strip, a second h1 above a second strip is two documents on one page,
          and it was measured costing 90 px of the fold before the first row. */}
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
            {headlineSentence(board, locale)}
            {seededLine ? ' ' : ''}
            {seededLine ? <span className="pacing-seeded">{seededLine}</span> : null}
          </p>
        </div>
        {/* The tablist holds tabs and nothing else. Read again is a command and
            not a view, and inside a role=tablist it made a reader count three
            tabs and find two. It sits beside the list, in the same group. */}
        <div className="pacing-views">
          <nav className="pacing-view-tabs" role="tablist" aria-label={pick(locale, 'Pacing views', 'תצוגות קצב')} onKeyDown={moveView}>
            <Button type="button" role="tab" id="pacing-tab-board" aria-controls="pacing-panel-board" data-pacing-view={BOARD}
                    tabIndex={view === BOARD ? 0 : -1} aria-selected={view === BOARD}
                    className={view === BOARD ? 'active' : ''} onClick={() => setView(BOARD)}>
              {pick(locale, 'Campaign pacing', 'קצב אספקה של הקמפיינים')}
            </Button>
            <Button type="button" role="tab" id="pacing-tab-ledger" aria-controls="pacing-panel-ledger" data-pacing-view={LEDGER}
                    tabIndex={view === LEDGER ? 0 : -1} aria-selected={view === LEDGER}
                    className={view === LEDGER ? 'active' : ''} onClick={() => setView(LEDGER)}>
              {pick(locale, 'Decision ledger', 'ספר ההחלטות')}
              {ledger.status === 'ready' && (ledger.payload.open_count + ledger.payload.accepted_count)
                ? (
                  <Figure className="pacing-open-count">
                    {ledger.payload.open_count + ledger.payload.accepted_count}
                  </Figure>
                )
                : null}
            </Button>
          </nav>
          <Button type="button" className="pacing-refresh" onClick={reload}>
            {pick(locale, 'Read again', 'קראו שוב')}
          </Button>
        </div>
      </div>

      {refusal ? (
        <div className="pacing-refusal" role="alert">
          <p>{refusal}</p>
          <div className="pacing-refusal-acts">
            {refusalOpen ? (
              <Button type="button" className="pacing-refusal-open" onClick={followRefusal}>
                {refusalOpen.kind === 'campaign'
                  ? pick(locale, 'Open that campaign', 'פתחו את הקמפיין')
                  : pick(locale, 'Open that record', 'פתחו את הרשומה')}
              </Button>
            ) : null}
            {/* A refusal caused by a stale read leaves the row on screen still
                offering the act that has just failed. The reload is offered and
                never taken: taking it would throw away what the reader typed
                into an open form, which is the defect a previous round closed. */}
            <Button type="button" onClick={() => { clearRefusal(); reload(); }}>
              {pick(locale, 'Read again', 'קראו שוב')}
            </Button>
            <Button type="button" onClick={clearRefusal}>{pick(locale, 'Dismiss', 'סגרו')}</Button>
          </div>
        </div>
      ) : null}

      {notice ? (
        <div className="pacing-notice" role="status">
          <p>{notice}</p>
          <div className="pacing-refusal-acts">
            {/* The act that reverses the act just announced. Its word, its
                transition and its reason are the ledger's, published on the
                answer to the write, and its title says what it does. */}
            {undoable ? (
              <Button type="button" className="pacing-undo"
                      title={pick(locale, undoable.undo.meaning_en, undoable.undo.meaning_he)}
                      onClick={() => onUndo(undoable)}>
                {pick(locale, undoable.undo.label_en, undoable.undo.label_he)}
              </Button>
            ) : null}
            <Button type="button" onClick={() => { setNotice(''); setUndoable(null); }}>
              {pick(locale, 'Dismiss', 'סגרו')}
            </Button>
          </div>
        </div>
      ) : null}

      {view === BOARD ? (
        <div id="pacing-panel-board" role="tabpanel" aria-labelledby="pacing-tab-board" tabIndex={0} aria-busy={board.status === 'loading'}>
          {board.status === 'loading' ? <p className="pacing-loading">{pick(locale, 'Reading the pacing board', 'קורא את לוח הקצב')}</p> : null}
          {board.status === 'failed' ? (
            <Failed locale={locale} onRetry={reload}
              en="The pacing board could not be read. What is missing is a failure, not an empty result."
              he="לא ניתן היה לקרוא את לוח הקצב. מה שחסר הוא כשל, לא תוצאה ריקה."
            />
          ) : null}
          {board.status === 'ready' ? (
            <PacingBoard
              payload={board.payload}
              locale={locale}
              canEdit={canEdit}
              editRefusal={editRefusal}
              busyId={busyId}
              onRaise={onRaise}
              onAccept={onAccept}
              onOpenMakeGood={openMakeGood}
              onOpenCampaign={onOpenCampaign ? openCampaign : null}
              focusCampaignId={focusCampaign}
              onFocused={clearFocus}
            />
          ) : null}
        </div>
      ) : null}

      {view === LEDGER ? (
        <div id="pacing-panel-ledger" role="tabpanel" aria-labelledby="pacing-tab-ledger" tabIndex={0} aria-busy={ledger.status === 'loading'}>
          {ledger.status === 'ready' ? (
            <MakeGoodLedger
              payload={ledger.payload}
              locale={locale}
              canEdit={canEdit}
              editRefusal={editRefusal}
              busyId={busyId}
              onMove={onMove}
              onOpenCampaign={openCampaign}
              focusMakeGoodId={focusMakeGood}
              onFocused={clearMakeGoodFocus}
            />
          ) : null}
          {ledger.status === 'failed' ? (
            <Failed locale={locale} onRetry={reload}
              en="The decision ledger could not be read. What is missing is a failure, not an empty result."
              he="לא ניתן היה לקרוא את ספר ההחלטות. מה שחסר הוא כשל, לא תוצאה ריקה."
            />
          ) : null}
          {ledger.status === 'loading' ? <p className="pacing-loading">{pick(locale, 'Reading the decision ledger', 'קורא את ספר ההחלטות')}</p> : null}
        </div>
      ) : null}
    </section>
  );
}
