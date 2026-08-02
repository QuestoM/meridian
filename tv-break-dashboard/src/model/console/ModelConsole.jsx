import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Numeric } from '../../shell/format';
import CandidatesPanel from './CandidatesPanel';
import CoveragePanel from './CoveragePanel';
import DriftPanel from './DriftPanel';
import GatesPanel from './GatesPanel';
import ProvenancePanel from './ProvenancePanel';
import TrainingPanel from './TrainingPanel';
import VersionsPanel from './VersionsPanel';
import { SECTIONS, SECTION_ROUTE, readConsole, readSection, recordVersion } from './console-api';
import { Absent } from './console-bits';
import { pick, t } from './console-words';
import './model-console.css';
import './model-console-panels.css';

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
      <div className="mc-header-marker">{t('console.marker', locale)}</div>
      <div className="mc-header-main">
        <div className="mc-header-identity">
          <h1>{t('console.title', locale)}</h1>
          <p>{t('console.subtitle', locale)}</p>
        </div>
        {onBack ? (
          <button type="button" className="mc-button" onClick={onBack}>
            {t('console.back', locale)}
          </button>
        ) : null}
      </div>
      {version.available ? (
        <div className="mc-header-strip">
          <div className="mc-header-version">
            <span className="mc-header-label">{t('header.version', locale)}</span>
            <strong dir="ltr"><Numeric>{version.name}</Numeric></strong>
            <code dir="ltr" className="mc-header-hash">{version.short}</code>
            {version.recorded ? (
              <span className="mc-chip mc-active">{t('header.recorded', locale)}</span>
            ) : (
              <button type="button" className="mc-link" onClick={onRecord} disabled={recording}>
                {t('header.not_recorded', locale)} - {t('header.record', locale)}
              </button>
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
              <button type="button" className="mc-link" onClick={onOpenRules}>
                {t('header.control_on_rules', locale)}
              </button>
            ) : (
              <small>{t('header.control_on_rules', locale)}</small>
            )}
          </div>
        </div>
      ) : (
        <Absent
          title={locale === 'en' ? 'No trained model on disk.' : 'אין בדיסק מודל מאומן.'}
          reason={pick(version, 'reason', locale)}
        />
      )}
    </header>
  );
}

function Rail({ section, onPick, locale }) {
  return (
    <nav className="mc-rail" aria-label={t('console.title', locale)}>
      {SECTIONS.map((id, index) => (
        <button
          type="button"
          key={id}
          className={`mc-rail-item ${section === id ? 'on' : ''}`}
          onClick={() => onPick(id)}
          aria-current={section === id ? 'page' : undefined}
        >
          <span>{t(`section.${id}`, locale)}</span>
          <kbd dir="ltr">{index + 1}</kbd>
        </button>
      ))}
    </nav>
  );
}

function Body({ section, state, locale, blocked, onRefresh, decideFor, onDecide, setDecideFor, refreshKey }) {
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
  if (section === 'coverage') return <CoveragePanel payload={payload} locale={locale} />;
  if (section === 'drift') return <DriftPanel payload={payload} locale={locale} />;
  if (section === 'candidates') {
    return (
      <CandidatesPanel
        payload={payload}
        locale={locale}
        onRefresh={onRefresh}
        onDecide={onDecide}
        refreshKey={refreshKey}
      />
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

export default function ModelConsole({ locale = 'he', onBack, onOpenRules }) {
  const [section, setSection] = useState('gates');
  const [refreshKey, setRefreshKey] = useState(0);
  const [recording, setRecording] = useState(false);
  const [decideFor, setDecideFor] = useState('');
  const [blocked, setBlocked] = useState(null);
  const head = useConsole(refreshKey);
  const body = useSection(section, refreshKey);
  const refresh = useCallback(() => setRefreshKey((key) => key + 1), []);

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
        setSection(next);
        event.preventDefault();
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // Pressing the verdict control moves the reader itself. Deriving the move
  // from a change in which candidate was picked looked equivalent and was not:
  // picking the same candidate twice is not a change, so the second press did
  // nothing at all. Measured on the shelf: open the form for a candidate, go
  // back to the shelf by the rail, press that candidate's control again, and
  // the screen stayed where it was.
  const decide = useCallback((candidate) => {
    setDecideFor(candidate.id);
    setSection('versions');
  }, []);

  // A section change starts at the top of that section. Without this, pressing
  // a control low down one section and landing on a shorter one leaves the
  // reader looking at blank space with the new screen scrolled off above.
  // Measured on the fifth candidate card: the verdict form opened and nothing
  // appeared to happen.
  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [section]);

  async function record() {
    setRecording(true);
    await recordVersion();
    setRecording(false);
    refresh();
  }

  const headPayload = useMemo(() => head.payload || {}, [head.payload]);
  const dir = locale === 'en' ? 'ltr' : 'rtl';

  return (
    <div className={`mc-console ${dir}`} dir={dir} lang={locale}>
      {head.status === 'refused' ? (
        <Absent title={t('state.refused', locale)} reason={head.detail} />
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
        <div className="mc-loading">{t('state.loading', locale)}</div>
      ) : (
        <Absent title={t('state.unreachable', locale)} reason={head.detail} />
      )}
      <div className="mc-layout">
        <Rail section={section} onPick={setSection} locale={locale} />
        <main className="mc-body">
          <p className="mc-route" dir="ltr">
            <code>{SECTION_ROUTE[section]}</code>
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
          />
        </main>
      </div>
    </div>
  );
}
