import React from 'react';

// Direction and isolation. This file is the only place in the dashboard that is
// allowed to state how a run of text relates to the direction around it, and
// verify-direction-rules.mjs fails the build when a surface states it anywhere
// else. A change to how figures, codes or names sit in a Hebrew line is a change
// to this file and to nothing else.
//
// THE DISTINCTION THIS FILE EXISTS TO HOLD
//
// A number, a date or a Latin identifier dropped into a Hebrew sentence needs
// its own internal order protected, or the bidi algorithm pulls the neighbouring
// digits and separators into it and prints the parts out of sequence. There are
// two ways to ask for that and they are not interchangeable.
//
// An OVERRIDE (the dir attribute, or the direction property) sets the base
// direction of the element it sits on. It fixes the internal order, which is
// wanted, and it also re-anchors the element's own alignment, which is not. On a
// block or a table cell that second effect is a bug: text-align: start resolves
// against the element's own direction, so a cell carrying dir="ltr" inside a
// Hebrew table aligns its content left while every neighbouring cell aligns
// right, and the column stops lining up. Measured on the shipped week-compare
// table: the row header sat 8px from the cell's right edge and the three numeric
// cells sat 321px, 320px and 212px from it.
//
// ISOLATION (unicode-bidi: isolate, or the U+2066..U+2069 characters) protects
// the run's internal order without touching which edge anything aligns to. It is
// the correct tool, and it is what every export below uses.
//
// The components render an inline span on purpose. Isolation on an inline box
// cannot re-anchor the alignment of the block that contains it, so the cell or
// paragraph keeps the document's direction and text-align: start keeps meaning
// "the reading edge". Never move these class names onto a td, div or li.

// Forced left-to-right internal order, for a run whose sequence is known to be
// left-to-right whatever the surrounding language: digits, dates, identifiers.
const FIGURE_CLASS = 'bidi-figure';
// The same order, for a machine-facing string that may be long enough to need a
// break inside itself rather than pushing its container wide.
const CODE_CLASS = 'bidi-code';
// Order inferred from the run's own first strong character, for a run that
// arrives as data and may be Hebrew or Latin.
const NAME_CLASS = 'bidi-name';

// U+2068 is the first-strong isolate: it infers the run's direction from the
// run's own first strong character, so one call is correct for a Hebrew channel
// name and for a Latin one, and correct in both locales. U+2069 pops it.
//
// Written as escapes on purpose. The characters render as nothing, so a literal
// pair in the source is invisible to review and to any editor that trims it.
const FIRST_STRONG_ISOLATE = '\u2068';
const POP_DIRECTIONAL_ISOLATE = '\u2069';

// Isolate a value that is being joined into a plain string rather than rendered
// as its own element: a heading built by concatenation, an aria-label, a title
// attribute, a toast. Prefer the components below wherever an element can go,
// because the CSS form needs no invisible characters in the data.
//
// Takes a numeral, a date, an identifier or a name. It does not take a phrase
// that already reads as a sentence in the surrounding language: isolating a
// Hebrew phrase inside a Hebrew line is a no-op at best, and a caller that
// isolates a value which already carries its own unit puts the unit in front of
// its own number.
export function isolate(value) {
  const text = String(value ?? '').trim();
  return text ? `${FIRST_STRONG_ISOLATE}${text}${POP_DIRECTIONAL_ISOLATE}` : '';
}

// A measured quantity, as the operator reads it: money, a count, a percentage, a
// duration, rating points, or the calendar date one of those is measured on.
export function Figure({ children, className, title }) {
  return (
    <span className={className ? `${FIGURE_CLASS} ${className}` : FIGURE_CLASS} title={title}>
      {children}
    </span>
  );
}

// A machine-facing string the operator is meant to read literally: a file path,
// a URL, an engine key, a job or version identifier, a log line, a column name.
export function Code({ children, className, title }) {
  return (
    <span className={className ? `${CODE_CLASS} ${className}` : CODE_CLASS} title={title}>
      {children}
    </span>
  );
}

// A name that arrives as data and may be written in either script: a channel, an
// advertiser, an agency, a programme, an uploaded file's title. The order comes
// from the name itself, so a Hebrew channel stays Hebrew inside an English line
// and a Latin brand stays Latin inside a Hebrew one.
export function Name({ children, className, title }) {
  return (
    <span className={className ? `${NAME_CLASS} ${className}` : NAME_CLASS} title={title}>
      {children}
    </span>
  );
}

// The direction a whole screen reads in, from the operator's locale.
export function documentDirection(locale) {
  return locale === 'he' ? 'rtl' : 'ltr';
}

// A DIRECTION ROOT is the one kind of element that SHOULD carry a direction
// override, because it is establishing the direction rather than escaping one.
// There are four of them and no more:
//
//   the application shell, which sets the direction the whole dashboard reads in;
//   a screen rendered before a locale exists, such as the Hebrew login;
//   an overlay or dialog rendered through a portal, which lands outside the
//   shell's subtree in the DOM and so inherits nothing from it;
//   a self-contained widget mounted into a host that is not the shell.
//
// The portal case is why this cannot simply be deleted everywhere. A portalled
// dialog with no direction root falls back to the document default and a Hebrew
// dialog renders left-to-right.
//
// Everything BELOW a root inherits from it and must not restate it. A section,
// a panel or a card writing dir={locale === 'he' ? 'rtl' : 'ltr'} is not a root:
// it is the disease this file exists to cure, and it should simply be deleted.
//
// Refs are forwarded because a root is often the element a widget measures,
// scrolls or focuses.
export const DirectionRoot = React.forwardRef(
  function DirectionRoot({ locale, as: Element = 'div', children, ...rest }, ref) {
    return (
      <Element dir={documentDirection(locale)} ref={ref} {...rest}>
        {children}
      </Element>
    );
  },
);

// The document itself is a direction root, and it is the one nobody was setting.
//
// index.html shipped <html lang="en"> with no dir at all, so the document
// declared an English left-to-right page for a product that is Hebrew and right
// to left. Two things follow from that and both are real. Assistive technology
// announces the whole page in the wrong language, which is not a detail for an
// operator using a screen reader in Hebrew. And the portal case named above
// resolves against the DOCUMENT default: a dialog rendered outside the shell's
// subtree with no root of its own inherits English left-to-right, which is
// exactly the failure the paragraph above warns about.
//
// The static file now ships he and rtl, matching the settings the plan
// fingerprint pins. This keeps it true when the operator uses the English
// toggle, so the two never disagree.
export function useDocumentDirection(locale) {
  React.useEffect(() => {
    const root = typeof document === 'undefined' ? null : document.documentElement;
    if (!root) return;
    root.lang = locale === 'he' ? 'he' : 'en';
    root.dir = documentDirection(locale);
  }, [locale]);
}

// A block of text whose language is not known until it arrives: a paragraph the
// model wrote, an operator's free-text note, a message from the server. Each
// block is its own direction root, taking both its order and its alignment from
// its own first strong character, so a Hebrew paragraph reads right to left and
// an English one beside it reads left to right.
//
// This is a deliberate override rather than an isolate. Isolation alone would
// order the paragraph correctly and still align it to the direction it
// inherited, which puts a Hebrew paragraph on the wrong edge of the thread.
//
// Use it only for text somebody else wrote. Interface copy already knows its
// language and needs nothing.
// Refs are forwarded for the same reason DirectionRoot forwards them: a block
// that establishes its own direction is often the element a widget measures or
// scrolls. The composer's mention highlight is exactly that case - it must
// scroll in step with the field it sits behind, and it must resolve its
// direction from the same first strong character the field resolves from, or
// the highlight parts company with the words it marks.
export const Prose = React.forwardRef(
  function Prose({ as: Element = 'p', children, className, ...rest }, ref) {
    return (
      <Element dir="auto" className={className} ref={ref} {...rest}>
        {children}
      </Element>
    );
  },
);
