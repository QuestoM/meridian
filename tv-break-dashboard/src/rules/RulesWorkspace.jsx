import React, { useEffect, useMemo, useRef, useState } from 'react';
import { CalendarClock, CalendarDays, Coins, ScrollText, SlidersHorizontal, Tv } from 'lucide-react';
import { pageText } from '../shell/format';
import { ANONYMOUS_SESSION, doorFor, fetchSession } from '../session.js';
import RestrictionsPage from './RestrictionsPage';
import LicencePage from './LicencePage';
import ChannelPage from './ChannelPage';
import PricingManager from './PricingManager';
import WorthOfASecond from './WorthOfASecond';
import CalendarEvents from './CalendarEvents';
import PlanningLevers from './settings-levers';
import { nextRulesSection } from './rules-lib';
import { Pressable } from '../studio/dom-controls';
import { Button } from '../studio/actions';
import './rules-workspace.css';
import './studio-ledger-rules.css';

// Rules is one destination holding the things that constrain a plan or a price.
// A restriction, a regulatory limit, a rate card and the two declarations that
// decide what every figure means are all authored records with a scope and an
// effect, so they are one family and one place, and the section is a control in
// the content rather than four more entries in the navigation.
//
// The section a person lands on comes from the job on their account, so the
// programming representative opens restrictions, the compliance owner opens the
// licence and the yield owner opens the rate card with no click at all.

const SECTIONS = [
  {
    id: 'restrictions', door: 'rules.restrictions', icon: ScrollText, en: 'Restrictions', he: 'הגבלות',
    detailEn: 'Author and price the rules applied to future plan runs.',
    detailHe: 'כתיבה ותמחור של הכללים שיחולו על ריצות התכנון הבאות.',
  },
  {
    id: 'licence', door: 'rules.licence', icon: CalendarClock, en: 'Licence', he: 'רישיון',
    detailEn: 'Review the regulatory limits that every plan must satisfy.',
    detailHe: 'בדיקת מגבלות הרישיון שכל תוכנית חייבת לקיים.',
  },
  {
    id: 'rate_card', door: 'rules.rate_card', icon: Coins, en: 'Rate card', he: 'מחירון',
    detailEn: 'Maintain the commercial assumptions used to value inventory.',
    detailHe: 'ניהול ההנחות המסחריות שלפיהן מחושב ערך המלאי.',
  },
  {
    id: 'calendar', door: null, icon: CalendarDays, en: 'Events calendar', he: 'לוח אירועים',
    detailEn: 'Record dated events that change demand, price or availability.',
    detailHe: 'תיעוד אירועים שמשנים ביקוש, מחיר או זמינות.',
  },
  {
    id: 'channel', door: null, icon: Tv, en: 'Channel & model', he: 'ערוץ ומודל',
    detailEn: 'Verify the channel and modelling declarations that scope every figure.',
    detailHe: 'אימות הצהרות הערוץ והמודל שמגדירות את התחולה של כל נתון.',
  },
  {
    id: 'levers', door: null, icon: SlidersHorizontal, en: 'Planning levers', he: 'מנופי תכנון',
    detailEn: 'Control the saved parameters used by the planning engine.',
    detailHe: 'שליטה בפרמטרים השמורים שמשמשים את מנוע התכנון.',
  },
];

const SECTION_BY_DOOR = Object.fromEntries(
  SECTIONS.filter((section) => section.door).map((section) => [section.door, section.id]),
);

function PlanningLeversTransportGate({ locale, loading, onRetry }) {
  return (
    <div className="rules-section">
      <section className="card rules-card" role={loading ? 'status' : 'alert'} aria-live="polite">
        <h2>{pageText(locale, 'Saved planning levers', 'מנופי התכנון השמורים')}</h2>
        <p className="rules-inline-error">
          {loading
            ? pageText(locale, 'Reading the saved settings…', 'קורא את ההגדרות השמורות…')
            : pageText(
              locale,
              'Saved settings are unavailable. No fallback values are shown or writable.',
              'ההגדרות השמורות אינן זמינות. ערכי ברירת מחדל אינם מוצגים ואינם ניתנים לכתיבה.',
            )}
        </p>
        {!loading && (
          <Button type="button" variant="outlined" onClick={onRetry}>
            {pageText(locale, 'Retry', 'ניסיון חוזר')}
          </Button>
        )}
      </section>
    </div>
  );
}

function sectionFromLocation() {
  if (typeof window === 'undefined') return '';
  const value = new URLSearchParams(window.location.search).get('rules');
  return SECTIONS.some((section) => section.id === value) ? value : '';
}

export default function RulesWorkspace(props) {
  const { locale, notify, onGlobalRefresh, showInternalNavigation = true } = props;
  const [session, setSession] = useState(ANONYMOUS_SESSION);
  const [active, setActive] = useState(sectionFromLocation());
  const tabsRef = useRef([]);

  useEffect(() => {
    let alive = true;
    fetchSession()
      .then(({ session: resolved }) => { if (alive) setSession(resolved); })
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    function syncFromAddress() {
      const requested = sectionFromLocation();
      setActive((currentActive) => nextRulesSection(currentActive, requested));
    }
    window.addEventListener('popstate', syncFromAddress);
    window.addEventListener('hashchange', syncFromAddress);
    return () => {
      window.removeEventListener('popstate', syncFromAddress);
      window.removeEventListener('hashchange', syncFromAddress);
    };
  }, []);

  const landing = useMemo(() => SECTION_BY_DOOR[doorFor(session)] || 'restrictions', [session]);
  const current = active || landing;
  const currentSection = SECTIONS.find((section) => section.id === current) || SECTIONS[0];

  function open(id) {
    if (!SECTIONS.some((section) => section.id === id)) return;
    setActive(id);
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      params.set('rules', id);
      const next = `${window.location.pathname}?${params.toString()}${window.location.hash}`;
      window.history.pushState({ workspace: 'governance', section: id }, '', next);
    }
  }

  function onTabKeyDown(event, index) {
    let next = index;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = SECTIONS.length - 1;
    else if (event.key === 'ArrowRight') next = (index + (locale === 'he' ? -1 : 1) + SECTIONS.length) % SECTIONS.length;
    else if (event.key === 'ArrowLeft') next = (index + (locale === 'he' ? 1 : -1) + SECTIONS.length) % SECTIONS.length;
    else return;
    event.preventDefault();
    open(SECTIONS[next].id);
    tabsRef.current[next]?.focus();
  }

  return (
    <section className="rules-workspace">
      <header className="rules-hero" aria-labelledby="governance-section-title">
        <div>
          <h1 id="governance-section-title">{locale === 'he' ? currentSection.he : currentSection.en}</h1>
          <p>
            {locale === 'he' ? currentSection.detailHe : currentSection.detailEn}
          </p>
        </div>
      </header>

      {showInternalNavigation && <nav className="rules-tabs" role="tablist" aria-label={pageText(locale, 'Governance sections', 'מדורי ממשל')}>
        {SECTIONS.map((section, index) => {
          const Icon = section.icon;
          return (
            <Pressable
              ref={(node) => { tabsRef.current[index] = node; }}
              key={section.id}
              id={`rules-tab-${section.id}`}
              type="button"
              role="tab"
              className={`rules-tab${current === section.id ? ' active' : ''}`}
              aria-selected={current === section.id}
              aria-controls={`rules-panel-${section.id}`}
              tabIndex={current === section.id ? 0 : -1}
              onClick={() => open(section.id)}
              onKeyDown={(event) => onTabKeyDown(event, index)}
            >
              <Icon size={18} strokeWidth={1.75} aria-hidden="true" />
              {locale === 'he' ? section.he : section.en}
            </Pressable>
          );
        })}
      </nav>}

      <div
        className="rules-active-panel"
        id={`rules-panel-${current}`}
        role="tabpanel"
        aria-labelledby={showInternalNavigation ? `rules-tab-${current}` : 'governance-section-title'}
        tabIndex={0}
      >
        {current === 'restrictions' && (
          <RestrictionsPage locale={locale} notify={notify} onGlobalRefresh={onGlobalRefresh} {...props} />
        )}
        {current === 'licence' && (
          <LicencePage locale={locale} session={session} notify={notify} />
        )}
        {current === 'rate_card' && (
          <div className="rules-section">
            <WorthOfASecond locale={locale} />
            <PricingManager copy={props.copy} locale={locale} notify={notify} onGlobalRefresh={onGlobalRefresh} embedded />
          </div>
        )}
        {current === 'calendar' && (
          <CalendarEvents
            locale={locale}
            notify={notify}
            onGlobalRefresh={onGlobalRefresh}
            onOpenRateCard={() => open('rate_card')}
            embedded
          />
        )}
        {current === 'channel' && (
          <ChannelPage locale={locale} session={session} notify={notify} onGlobalRefresh={onGlobalRefresh} />
        )}
        {current === 'levers' && (props.settingsAvailable
          ? <PlanningLevers {...props} />
          : (
            <PlanningLeversTransportGate
              locale={locale}
              loading={props.settingsLoading}
              onRetry={onGlobalRefresh}
            />
          ))}
      </div>
    </section>
  );
}
