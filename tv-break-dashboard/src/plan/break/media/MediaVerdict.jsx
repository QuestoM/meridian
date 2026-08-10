import React from 'react';
import { pageText } from '../../../shell/format';
import './media-verdict.css';

// The technical verdict on one commercial's own FILE, printed ON the spot row
// rather than behind a click, which is what JS-8 asks for: a corrupt file should
// be obvious before it airs, and a fault you have to go looking for is a fault
// that ships.
//
// THREE STATES AND NEVER A FOURTH, and the middle one is the whole point.
// `verified` means a real file was inspected and matched. `failed` means it was
// inspected and did not. `unavailable` means NOBODY HAS INSPECTED IT, which is
// neither of the others: it has not been cleared, so it must not read as clean,
// and it has not been found wrong, so it must not read as broken. Today every
// spot is unavailable, because data/media_assets.csv is header-only and nothing
// in this product observes a media file.
//
// The dot is deliberately quiet for the unavailable case and loud only for a
// measured failure. If absence shouted, every row would shout, and a board where
// everything is flagged tells a reader nothing.

const WORDS = {
  verified: ['File checked', 'הקובץ נבדק'],
  failed: ['File fails', 'הקובץ נכשל'],
  unavailable: ['File not checked', 'הקובץ לא נבדק'],
};

const FACT_WORDS = {
  duration: ['length', 'אורך'],
  container: ['container', 'מכל'],
  codec: ['codec', 'קודק'],
  frame_rate: ['frame rate', 'קצב פריימים'],
  frame_shape: ['frame shape', 'צורת פריים'],
  audio: ['audio', 'שמע'],
  loudness: ['loudness', 'עוצמת שמע'],
  approval: ['approval', 'אישור'],
};

// What the reader is told when they rest on the mark. A failure names WHICH fact
// failed, because "this file is wrong" is not something anyone can act on.
export function verdictTitle(verdict, locale) {
  const say = (en, he) => pageText(locale, en, he);
  if (!verdict) return say('The file was not checked.', 'הקובץ לא נבדק.');
  if (verdict.state === 'failed') {
    const broken = Object.keys(verdict.facts || {})
      .filter((name) => verdict.facts[name].state === 'failed')
      .map((name) => say(FACT_WORDS[name]?.[0] || name, FACT_WORDS[name]?.[1] || name));
    const reasons = Object.values(verdict.facts || {})
      .filter((fact) => fact.state === 'failed')
      .map((fact) => (locale === 'he' ? fact.reason_he : fact.reason))
      .filter(Boolean);
    return `${say('The file fails on', 'הקובץ נכשל ב')}: ${broken.join(', ')}. ${reasons.join(' ')}`;
  }
  if (verdict.state === 'verified') {
    return say('The file was checked and matches its booking.', 'הקובץ נבדק ותואם את ההזמנה.');
  }
  return (locale === 'he' ? verdict.reason_he : verdict.reason)
    || say('The file was not checked.', 'הקובץ לא נבדק.');
}

export default function MediaVerdict({ verdict, locale }) {
  const state = verdict?.state || 'unavailable';
  const [en, he] = WORDS[state] || WORDS.unavailable;
  return (
    <span
      className={`media-verdict media-verdict-${state}`}
      title={verdictTitle(verdict, locale)}
      aria-label={verdictTitle(verdict, locale)}
    >
      <span className="media-verdict-dot" aria-hidden="true" />
      <span className="media-verdict-word">{pageText(locale, en, he)}</span>
    </span>
  );
}

export function MediaLockNotice({ media, locale }) {
  if (!media || !media.blocks_lock) return null;
  const houses = Array.isArray(media.blocking_house_numbers) ? media.blocking_house_numbers : [];
  return (
    <p className="media-lock-refusal" role="alert">
      {pageText(
        locale,
        `This pod cannot be locked. Measured media verification failed for House Number: ${houses.join(', ') || 'unknown'}.`,
        `לא ניתן לנעול את התוכן. אימות המדיה המדוד נכשל עבור House Number: ${houses.join(', ') || 'לא ידוע'}.`,
      )}
    </p>
  );
}
