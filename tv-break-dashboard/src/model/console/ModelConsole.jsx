import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Code, DirectionRoot, Prose, documentDirection } from '../../shell/bidi';
import { Numeric } from '../../shell/format';
import CandidateBoard from '../candidates/CandidateBoard.jsx';
import CandidatesPanel from './CandidatesPanel';
import CoveragePanel from './CoveragePanel';
import DriftPanel from './DriftPanel';
import GatesPanel from './GatesPanel';
import ProvenancePanel from './ProvenancePanel';
import TrainingPanel from './TrainingPanel';
import VersionsPanel from './VersionsPanel';
import { SECTIONS, SECTION_ROUTE, readConsole, readSection, recordVersion } from './console-api';
import { Absent } from './console-bits';
import { canEditReason, pick, t } from './console-words';
import { Pressable } from '../../studio/dom-controls';
import './model-console.css';
import './model-console-panels.css';
import './studio-ledger-model.css';

// The model console: a different shell for the company side of the line.
//
// It is a different shell rather than a page inside the operator's, because the
// two sides must never be confusable. The permanent marker in the header says
// which side this is, every way out is a control that names its own
// destination, and no operator surface reaches this component at all.
//
// The header answers the first question with zero clicks: which model version
// is in force, what its gates decided, whether it is recorded, and whether runs
// are consuming the audience model. Everything below is one of the five
// sections the specification names, plus training.

const SECTION_KEYS = {
  1: 'gates', 2: 'coverage', 3: 'drift', 4: 'candidates',
  5: 'training', 6: 'versions', 7: 'provenance',
};

const MODEL_SECTION_PARAM = 'modelSection';

function sectionFromLocation() {
  if (typeof window === 'undefined') return 'gates';
  const requested = new URLSearchParams(window.location.search).get(MODEL_SECTION_PARAM);
  return SECTIONS.includes(requested) ? requested : 'gates';
}

function useConsole(refreshKey) {
  const [state, setState] = useState({ status: 'loading', payload: null, detail: '' });
  useEffect(() => {
    let active = true;
    readConsole().then((result) => {
      if (active) setState(result);
    });
    return () => { active = false; };
  }, [refreshKey]);
  return state;
}

function useSection(section, refreshKey) {
  const [state, setState] = useState({ status: 'loading', payload: null, detail: '' });
  useEffect(() => {
    let active = true;
    setState({ status: 'loading', payload: null, detail: '' });
    readSection(section).then((result) => {
      if (active) setState(result);
    });
    return () => { active = false; };
  }, [section, refreshKey]);
  return state;
}

function activationLabel(activation, locale) {
  if (!activation || !activation.available) return t('header.activation_off', locale);
  if (activation.state === 'on') return t('header.activation_on', locale);
  if (activation.state === 'on_no_artifact') return t('header.activation_no_artifact', locale);
  return t('header.activation_off', locale);
}

function Header({ payload, locale, onRecord, onBack, onOpenRules, recording }) {
  const version = payload.model_version || {};
  const counts = payload.gate_counts || {};
  const activation = payload.activation || {};
  return (
    <header className="mc-header">
      <div className="mc-header-main">
        <div className="mc-header-identity">
          <div className="mc-header-title-row">
            <h1>{t('console.title', locale)}</h1>
            <span className="mc-header-marker">{t('console.marker', locale)}</span>
          </div>
          <p>{t('console.subtitle', locale)}</p>
        </div>
        {onBack ? (
          <Pressable type="button" className="mc-button" onClick={onBack}>
            {t('console.back', locale)}
          </Pressable>
        ) : null}
      </div>
      {version.available ? (
        <div className="mc-header-strip">
          <div className="mc-header-version">
            <span className="mc-header-label">{t('header.version', locale)}</span>
            <strong><Numeric>{version.name}</Numeric></strong>
            <code className="mc-header-hash"><Code>{version.short}</Code></code>
            {version.recorded ? (
              <span className="mc-chip mc-active">{t('header.recorded', locale)}</span>
            ) : (
              <Pressable type="button" className="mc-link" onClick={onRecord} disabled={recording}>
                {t('header.not_recorded', locale)} - {t('header.record', locale)}
              </Pressable>
            )}
          </div>
          <div className="mc-header-counts">
            {(payload.gate_states || []).map((state) => (
              <span className={`mc-count mc-${state.id}`} key={state.id}>
                <strong><Numeric>{String(counts[state.id] ?? 0)}</Numeric></strong>
                <small>{locale === 'en' ? state.en : state.he}</small>
              </span>
            ))}
          </div>
          {/*
            These counts do not move between trainings, and a count that never
            moves reads as a stuck system unless the screen says when it was
            last decided. Measured on the shipped console: the header carried a
            version date but nothing tied it to the counts beside it, so a
            reader had no way to know the 3/5/5/0 below was current rather than
            frozen. This line ties them together in words, from the same
            ``version.name`` the header already computes.
          */}
          <p className="mc-header-counts-note">
            {t('header.gates_measured_at', locale)}{' '}
            <Numeric>{version.name}</Numeric>
          </p>
          {/*
            The activation mirror, and the way to the switch it names.

            The console shows whether runs are consuming the audience model and
            carries no control for it, because throwing that switch changes a
            run and its home is Rules. That makes the sentence naming Rules a
            promise, so it is the control that keeps it rather than a note about
            it. Measured on the live DOM before this: this block held zero
            controls and the whole header held one, so the steward who read
            where the switch lives had no way to go there.

            It renders as a plain note only where there is no host to navigate,
            which is not a path the product mounts: the bridge passes the
            handler and the back control together, and a test asserts it does.
          */}
          <div className="mc-header-activation">
            <span className="mc-header-label">{t('header.activation', locale)}</span>
            <span className={`mc-chip ${activation.state === 'on' ? 'mc-active' : 'mc-no_contrast'}`}>
              {activationLabel(activation, locale)}
            </span>
            {onOpenRules ? (
              <Pressable type="button" className="mc-link" onClick={onOpenRules}>
                {t('header.control_on_rules', locale)}
              </Pressable>
            ) : (
              <small>{t('header.control_on_rules', locale)}</small>
            )}
          </div>
          {/*
            Who may run a rebuild, read from the same wall that would refuse the
            training POST, not asserted. ``can_edit`` absent means the payload
            predates this stamp and the line renders nothing rather than a
            guess; that only happens against a synthetic body in a test.
          */}
          {payload.can_edit === undefined ? null : (
            <div className="mc-header-permission">
              <span className="mc-header-label">{t('header.who_may_train', locale)}</span>
              {/*
                Prose, because the refusing branch is the wall's own sentence
                rather than interface copy: canEditReason falls back to whatever
                the server sent when it has no English for it, so the line can
                arrive in either language and has to take its direction from
                itself.
              */}
              <Prose as="span">
                {payload.can_edit
                  ? t('header.can_edit_yes', locale)
                  : canEditReason(payload.can_edit_reason, locale)}
              </Prose>
            </div>
          )}
        </div>
      ) : (
        <Absent
          title={locale === 'en' ? 'No trained model is available.' : 'אין מודל מאומן זמין.'}
          reason={pick(version, 'reason', locale)}
        />
      )}
    </header>
  );
}

function Rail({ section, onPick, locale }) {
  const tabsRef = useRef([]);

  function onKeyDown(event, index) {
    let next = index;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = SECTIONS.length - 1;
    else if (event.key === 'ArrowDown' || event.key === 'ArrowRight') next = (index + 1) % SECTIONS.length;
    else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') next = (index - 1 + SECTIONS.length) % SECTIONS.length;
    else return;
    event.preventDefault();
    onPick(SECTIONS[next]);
    tabsRef.current[next]?.focus();
  }

  return (
    <nav className="mc-rail" role="tablist" aria-orientation="vertical" aria-label={t('console.title', locale)}>
      {SECTIONS.map((id, index) => (
        <Pressable
          ref={(node) => { tabsRef.current[index] = node; }}
          type="button"
          key={id}
          id={`model-tab-${id}`}
          role="tab"
          className={`mc-rail-item ${section === id ? 'on' : ''}`}
          onClick={() => onPick(id)}
          onKeyDown={(event) => onKeyDown(event, index)}
          aria-selected={section === id}
          aria-controls={`model-panel-${id}`}
          tabIndex={section === id ? 0 : -1}
        >
          <span>{t(`section.${id}`, locale)}</span>
          <kbd><Numeric>{index + 1}</Numeric></kbd>
        </Pressable>
      ))}
    </nav>
  );
}

function Body({ section, state, locale, blocked, onRefresh, decideFor, onDecide, setDecideFor,
  refreshKey, onOpenEvents }) {
  if (state.status === 'loading') {
    return <div className="mc-loading">{t('state.loading', locale)}</div>;
  }
  if (state.status === 'refused') {
    return <Absent title={t('state.refused', locale)} reason={state.detail} />;
  }
  if (state.status !== 'ok') {
    return <Absent title={t('state.unreachable', locale)} reason={state.detail} />;
  }
  const payload = state.payload || {};
  if (section === 'gates') return <GatesPanel payload={payload} blocked={blocked} locale={locale} />;
  if (section === 'coverage') {
    return <CoveragePanel payload={payload} locale={locale} onOpenEvents={onOpenEvents} />;
  }
  if (section === 'drift') return <DriftPanel payload={payload} locale={locale} />;
  if (section === 'candidates') {
    // The board first, then the shelf. They answer different questions and the
    // order says which one to trust: the shelf reports each artifact's own
    // held-out figures, and on this tree those come from different splits, so
    // two rows of it compare two experiments rather than two models. The board
    // re-scores every artifact on one identical set of breaks. It was built
    // across ten critic rounds and, until this mount, no file in the tree
    // imported it, so it shipped to nobody.
    return (
      <>
        <CandidateBoard locale={locale} />
        <CandidatesPanel
          payload={payload}
          locale={locale}
          onRefresh={onRefresh}
          onDecide={onDecide}
          refreshKey={refreshKey}
        />
      </>
    );
  }
  if (section === 'training') return <TrainingPanel payload={payload} locale={locale} onRefresh={onRefresh} />;
  if (section === 'provenance') return <ProvenancePanel payload={payload} locale={locale} />;
  return (
    <VersionsPanel
      payload={payload}
      locale={locale}
      onRefresh={onRefresh}
      openForm={decideFor}
      onCloseForm={() => setDecideFor('')}
    />
  );
}

export default function ModelConsole({ locale = 'he', onBack, onOpenRules, onOpenEvents }) {
  const [section, setSection] = useState(sectionFromLocation);
  const [refreshKey, setRefreshKey] = useState(0);
  const [recording, setRecording] = useState(false);
  const [decideFor, setDecideFor] = useState('');
  const [blocked, setBlocked] = useState(null);
  const bodyRef = useRef(null);
  const head = useConsole(refreshKey);
  const body = useSection(section, refreshKey);
  const refresh = useCallback(() => setRefreshKey((key) => key + 1), []);

  const pickSection = useCallback((next, historyMode = 'push') => {
    if (!SECTIONS.includes(next)) return;
    setDecideFor('');
    if (next === section) return;
    setSection(next);
    if (typeof window === 'undefined') return;
    const params = new URLSearchParams(window.location.search);
    params.set(MODEL_SECTION_PARAM, next);
    const url = `${window.location.pathname}?${params.toString()}${window.location.hash}`;
    if (historyMode === 'replace') window.history.replaceState({ workspace: 'model', section: next }, '', url);
    else window.history.pushState({ workspace: 'model', section: next }, '', url);
  }, [section]);

  useEffect(() => {
    function syncFromAddress() {
      setSection(sectionFromLocation());
      setDecideFor('');
    }
    window.addEventListener('popstate', syncFromAddress);
    return () => window.removeEventListener('popstate', syncFromAddress);
  }, []);

  // The blocked register is read once and shown beside the gates it explains,
  // so a "no contrast" verdict on the gate table already says what would end it
  // rather than sending the reader to another section to find out.
  useEffect(() => {
    let active = true;
    readSection('coverage').then((result) => {
      if (!active || result.status !== 'ok') return;
      const rows = (result.payload || {}).blocked || [];
      setBlocked(Object.fromEntries(rows.map((row) => [row.gate_id, row])));
    });
    return () => { active = false; };
  }, [refreshKey]);

  useEffect(() => {
    function onKey(event) {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const tag = String(event.target && event.target.tagName).toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
      const next = SECTION_KEYS[event.key];
      if (next) {
        pickSection(next);
        event.preventDefault();
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [pickSection]);

  // Pressing the verdict control moves the reader itself. Deriving the move
  // from a change in which candidate was picked looked equivalent and was not:
  // picking the same candidate twice is not a change, so the second press did
  // nothing at all. Measured on the shelf: open the form for a candidate, go
  // back to the shelf by the rail, press that candidate's control again, and
  // the screen stayed where it was.
  const decide = useCallback((candidate) => {
    pickSection('versions');
    setDecideFor(candidate.id);
  }, [pickSection]);

  // A section change starts at the top of that section. Without this, pressing
  // a control low down one section and landing on a shorter one leaves the
  // reader looking at blank space with the new screen scrolled off above.
  // Measured on the fifth candidate card: the verdict form opened and nothing
  // appeared to happen.
  useEffect(() => {
    bodyRef.current?.scrollTo?.({ top: 0 });
  }, [section]);

  async function record() {
    setRecording(true);
    await recordVersion();
    setRecording(false);
    refresh();
  }

  const headPayload = useMemo(() => head.payload || {}, [head.payload]);
  // This surface is a shell of its own rather than a page inside the operator's,
  // so it is one of the few elements in the product that SHOULD state a
  // direction: nothing above it in the tree establishes one for the company
  // side. DirectionRoot is that statement, and it derives the value from the
  // same helper the app shell uses so the two sides cannot drift apart.
  const dir = documentDirection(locale);

  return (
    <DirectionRoot locale={locale} className={`mc-console ${dir}`} lang={locale}>
      {head.status === 'refused' ? (
        <div className="mc-state-page"><h1>{t('console.title', locale)}</h1><Absent title={t('state.refused', locale)} reason={head.detail} /></div>
      ) : head.status === 'ok' ? (
        <Header
          payload={headPayload}
          locale={locale}
          onRecord={record}
          onBack={onBack}
          onOpenRules={onOpenRules}
          recording={recording}
        />
      ) : head.status === 'loading' ? (
        <div className="mc-state-page" aria-busy="true"><h1>{t('console.title', locale)}</h1><div className="mc-loading" role="status">{t('state.loading', locale)}</div></div>
      ) : (
        <div className="mc-state-page"><h1>{t('console.title', locale)}</h1><Absent title={t('state.unreachable', locale)} reason={head.detail} /></div>
      )}
      <div className="mc-layout">
        <Rail section={section} onPick={pickSection} locale={locale} />
        <main ref={bodyRef} className="mc-body" id={`model-panel-${section}`} role="tabpanel" aria-labelledby={`model-tab-${section}`} tabIndex={0}>
          <p className="mc-route">
            <code><Code>{SECTION_ROUTE[section]}</Code></code>
          </p>
          <Body
            section={section}
            state={body}
            locale={locale}
            blocked={blocked}
            onRefresh={refresh}
            decideFor={decideFor}
            onDecide={decide}
            setDecideFor={setDecideFor}
            refreshKey={refreshKey}
            onOpenEvents={onOpenEvents}
          />
        </main>
      </div>
    </DirectionRoot>
  );
}
