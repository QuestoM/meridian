import React, { useEffect, useMemo, useRef, useState } from 'react';
import { CalendarClock, CalendarDays, Coins, ScrollText, SlidersHorizontal, Tv } from 'lucide-react';
import { pageText } from '../shell/format';
import { ANONYMOUS_SESSION, doorFor, fetchSession } from '../session.js';
import { word } from '../vocabulary.js';
import RestrictionsPage from './RestrictionsPage';
import LicencePage from './LicencePage';
import ChannelPage from './ChannelPage';
import PricingManager from './PricingManager';
import WorthOfASecond from './WorthOfASecond';
import CalendarEvents from './CalendarEvents';
import PlanningLevers from './settings-levers';
import { nextRulesSection } from './rules-lib';
import './rules-workspace.css';

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
  { id: 'restrictions', door: 'rules.restrictions', icon: ScrollText, en: 'Restrictions', he: 'הגבלות' },
  { id: 'licence', door: 'rules.licence', icon: CalendarClock, en: 'The licence', he: 'הרישיון' },
  { id: 'rate_card', door: 'rules.rate_card', icon: Coins, en: 'The rate card', he: 'כרטיס התעריפים' },
  { id: 'calendar', door: null, icon: CalendarDays, en: 'The calendar', he: 'לוח האירועים' },
  { id: 'channel', door: null, icon: Tv, en: 'Channel and model', he: 'ערוץ ומודל' },
  { id: 'levers', door: null, icon: SlidersHorizontal, en: 'Planning levers', he: 'מנופי התכנון' },
];

const SECTION_BY_DOOR = Object.fromEntries(
  SECTIONS.filter((section) => section.door).map((section) => [section.door, section.id]),
);

function sectionFromLocation() {
  if (typeof window === 'undefined') return '';
  const value = new URLSearchParams(window.location.search).get('rules');
  return SECTIONS.some((section) => section.id === value) ? value : '';
}

export default function RulesWorkspace(props) {
  const { locale, notify, onGlobalRefresh } = props;
  const [session, setSession] = useState(ANONYMOUS_SESSION);
  const [active, setActive] = useState(sectionFromLocation());
  // The last ?rules value this render has already reconciled against, so a
  // query this workspace's own open() just wrote is never re-applied a
  // second time and only a change from elsewhere moves the section on its own.
  const seenQuery = useRef(sectionFromLocation());

  useEffect(() => {
    let alive = true;
    fetchSession()
      .then(({ session: resolved }) => { if (alive) setSession(resolved); })
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  // Read the query on every render, not only the one this component mounted
  // with. A route elsewhere in the shell can rewrite ?rules without ever
  // remounting this workspace (the old Pricing bookmark redirect is exactly
  // this), and the browser's own back and forward buttons do the same, so
  // following the query only at mount left the section stuck on whatever tab
  // was open when that redirect landed while the address bar claimed another.
  const queryNow = sectionFromLocation();
  if (queryNow !== seenQuery.current) {
    seenQuery.current = queryNow;
    const moved = nextRulesSection(active, queryNow);
    if (moved !== active) setActive(moved);
  }

  const landing = useMemo(() => SECTION_BY_DOOR[doorFor(session)] || 'restrictions', [session]);
  const current = active || landing;

  function open(id) {
    setActive(id);
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      params.set('rules', id);
      const next = `${window.location.pathname}?${params.toString()}${window.location.hash}`;
      window.history.replaceState(null, '', next);
    }
  }

  return (
    <section className="rules-workspace">
      <div className="rules-hero">
        <div>
          <span className="rules-kicker">{word('place.rules', locale)}</span>
          <h1>
            {pageText(
              locale,
              'What constrains a plan and what prices it',
              'מה מגביל תוכנית ומה מתמחר אותה',
            )}
          </h1>
          <p>
            {pageText(
              locale,
              'Every rule here is an authored record with a scope, an effect and somebody who asked for it. Each one states what it costs before it is saved.',
              'כל כלל כאן הוא רשומה כתובה עם תחולה, השפעה ומי שביקש אותה. כל אחת מהן מציגה את העלות לפני השמירה.',
            )}
          </p>
        </div>
      </div>

      <nav className="rules-tabs" aria-label={pageText(locale, 'Rules sections', 'מדורי הכללים')}>
        {SECTIONS.map((section) => {
          const Icon = section.icon;
          return (
            <button
              key={section.id}
              type="button"
              className={`rules-tab${current === section.id ? ' active' : ''}`}
              aria-current={current === section.id ? 'page' : undefined}
              onClick={() => open(section.id)}
            >
              <Icon size={14} aria-hidden="true" />
              {locale === 'he' ? section.he : section.en}
            </button>
          );
        })}
      </nav>

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
      {/* An event dated on the calendar shapes what a break is worth, which is
          why it is a rule and not a topic. Its own page is still reachable at
          its old address, so nothing that worked before this section stopped. */}
      {current === 'calendar' && (
        <CalendarEvents
          locale={locale}
          notify={notify}
          onGlobalRefresh={onGlobalRefresh}
          onOpenRateCard={() => open('rate_card')}
        />
      )}
      {current === 'channel' && (
        <ChannelPage locale={locale} session={session} notify={notify} onGlobalRefresh={onGlobalRefresh} />
      )}
      {current === 'levers' && <PlanningLevers {...props} />}
    </section>
  );
}
