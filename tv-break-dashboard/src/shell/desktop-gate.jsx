import React, { useEffect, useState } from 'react';
import { DirectionRoot } from './bidi';
import { Button } from '../studio/actions';
import { KairosMark } from './kairos-icons';

export const MIN_DESKTOP_WIDTH = 1200;

function media(query) {
  return typeof window !== 'undefined' && typeof window.matchMedia === 'function'
    ? window.matchMedia(query)
    : null;
}

function supportedDesktopSnapshot() {
  if (typeof window === 'undefined' || typeof document === 'undefined') return true;

  const layoutWidth = Math.max(0, document.documentElement.clientWidth || window.innerWidth || 0);
  if (layoutWidth >= MIN_DESKTOP_WIDTH) return true;

  // Browser zoom can reduce CSS viewport width even on a wide desktop. There
  // is no standard zoom signal, so use the conservative evidence available:
  // a fine hover pointer, no coarse primary pointer, a physically wide screen,
  // and a wide outer browser window. Phones and tablets fail these tests in
  // both orientations; a deliberately narrow desktop window still gets the
  // gate because its outer window is narrow too.
  const finePointer = Boolean(media('(hover: hover) and (pointer: fine)')?.matches);
  const coarsePointer = Boolean(media('(pointer: coarse)')?.matches);
  const availableWidth = Number(window.screen?.availWidth || window.screen?.width || 0);
  const outerWidth = Number(window.outerWidth || 0);
  const zoomRatio = layoutWidth > 0 ? outerWidth / layoutWidth : 0;
  const zoomEvidence = outerWidth >= MIN_DESKTOP_WIDTH
    && layoutWidth > 0
    && zoomRatio >= 1.12;

  return finePointer
    && !coarsePointer
    && availableWidth >= MIN_DESKTOP_WIDTH
    && zoomEvidence;
}

export function useDesktopSupport() {
  const [supported, setSupported] = useState(supportedDesktopSnapshot);

  useEffect(() => {
    const queries = [
      media(`(min-width: ${MIN_DESKTOP_WIDTH}px)`),
      media('(hover: hover) and (pointer: fine)'),
      media('(pointer: coarse)'),
    ].filter(Boolean);
    let frame = 0;
    let demotion = 0;
    const measure = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        if (supportedDesktopSnapshot()) {
          // Promotion is immediate; a pending demotion is cancelled, because
          // the dip it was timing has already ended.
          window.clearTimeout(demotion);
          demotion = 0;
          setSupported(true);
          return;
        }
        // Demotion waits until the unsupported state PERSISTS. The gate
        // replaces the whole tree, so gating on a momentary dip below the
        // threshold — a window drag passing through, a window-manager
        // animation, a screenshot tool resizing the surface for one frame —
        // unmounts the operator's entire workspace and remounts it blank. A
        // real phone or tablet is unsupported from the first snapshot (the
        // initial state, which does not wait), and a genuinely narrowed
        // window is still unsupported when the timer looks again.
        if (!demotion) {
          demotion = window.setTimeout(() => {
            demotion = 0;
            setSupported(supportedDesktopSnapshot());
          }, 600);
        }
      });
    };

    window.addEventListener('resize', measure, { passive: true });
    window.addEventListener('orientationchange', measure, { passive: true });
    queries.forEach((query) => query.addEventListener?.('change', measure));
    measure();
    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(demotion);
      window.removeEventListener('resize', measure);
      window.removeEventListener('orientationchange', measure);
      queries.forEach((query) => query.removeEventListener?.('change', measure));
    };
  }, []);

  return supported;
}

function initialLocale() {
  try {
    const saved = window.localStorage.getItem('kairos.locale');
    if (saved === 'en' || saved === 'he') return saved;
  } catch {
    // The language toggle remains available without storage.
  }
  if (typeof document !== 'undefined' && document.documentElement.lang === 'en') return 'en';
  return typeof navigator !== 'undefined' && String(navigator.language || '').toLowerCase().startsWith('he')
    ? 'he'
    : 'en';
}

const GATE_COPY = {
  en: {
    eyebrow: 'Kairos · Revenue operations',
    title: 'Continue on a desktop',
    body: 'Kairos is a high-density broadcast planning console. Open it in a desktop browser so timelines, commercial records, and decision evidence remain precise and safe to operate.',
    requirement: 'A desktop viewport of at least 1,200 pixels is required.',
    switchLanguage: 'עברית',
  },
  he: {
    eyebrow: 'Kairos · ניהול הכנסות',
    title: 'ממשיכים ממחשב שולחני',
    body: 'Kairos היא סביבת תכנון צפופה לשידורים. פתחו אותה בדפדפן במחשב שולחני כדי שצירי הזמן, הרשומות המסחריות והראיות להחלטות יישארו מדויקים ובטוחים לתפעול.',
    requirement: 'נדרש חלון דסקטופ ברוחב 1,200 פיקסלים לפחות.',
    switchLanguage: 'English',
  },
};

export default function DesktopGate() {
  const [locale, setLocale] = useState(initialLocale);
  const words = GATE_COPY[locale];

  useEffect(() => {
    document.documentElement.lang = locale;
    document.documentElement.dir = locale === 'he' ? 'rtl' : 'ltr';
    try {
      window.localStorage.setItem('kairos.locale', locale);
    } catch {
      // The selected language still applies for this visit.
    }
  }, [locale]);

  return (
    <DirectionRoot as="main" locale={locale} className="desktop-gate" lang={locale}>
      <section className="desktop-gate-card" aria-labelledby="desktop-gate-title">
        <KairosMark className="desktop-gate-mark" size={48} title="Kairos" />
        <p className="desktop-gate-eyebrow">{words.eyebrow}</p>
        <h1 id="desktop-gate-title">{words.title}</h1>
        <p>{words.body}</p>
        <p className="desktop-gate-requirement">{words.requirement}</p>
        <Button variant="contained" onClick={() => setLocale((current) => (current === 'he' ? 'en' : 'he'))}>
          {words.switchLanguage}
        </Button>
      </section>
    </DirectionRoot>
  );
}
