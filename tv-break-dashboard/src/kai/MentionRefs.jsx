import React from 'react';
import { CircleCheck, History, TriangleAlert, X } from 'lucide-react';
import { Name, Prose } from '../shell/bidi';
import { pageText } from '../shell/surface-helpers';
import { iconFor } from './MentionPicker';
import { chipRuns } from './mention-refs';
import './mention-refs.css';

// WHAT A MENTION BOUND TO, ON SCREEN.
//
// The audit's finding against page_context is not that it is badly built. It is
// well built. The finding is that THE BINDING IS INVISIBLE: nothing tells the
// operator that "it" resolved, or to what. A typed reference that only the model
// can see would repeat that defect one layer down, so this file exists to make
// the binding a thing you can look at.
//
// Two moments, and they say different things because different things are true
// at each.
//
// BEFORE SENDING, the strip says what is ATTACHED: the kind, the name, and the
// rail glyph for the kind. It does not say what the object contains, because
// nothing has been read yet. Resolution is at SEND time and never at insertion
// time, and for a stronger reason here than in either reference product: a RUN
// rewrites the plan underneath the operator, so an insertion-time snapshot would
// let a pre-run figure be quoted as current.
//
// AFTER SENDING, the strip under the question says what each reference CAME BACK
// AS, in the four states the server resolved them into. Silent drop is forbidden
// here: a dead reference that vanished would leave a Hebrew label in the
// question with no data behind it, and the rule that every figure names its
// basis would push the model to answer from the label. That is fabrication.
//
// The words are the server's. Every state word and every kind word comes down on
// the payload from assistant_mentions_words.py, where each is read from a module
// that already ships it. Nothing here translates anything.

const STATE_GLYPHS = {
  resolved: CircleCheck,
  changed: History,
  gone: X,
  unavailable: TriangleAlert,
};

function stateWord(mention, locale) {
  const word = locale === 'he' ? mention.state_he : mention.state_en;
  return word || '';
}

// One chip. An inline span, always: bidi.jsx states it plainly, isolation on a
// block re-anchors alignment, so a chip built as a div would left-align the line
// it sits in inside the Hebrew dock. The glyph is a sibling span inside the
// Name, and the whole thing stays inline.
function Chip({ mention, locale }) {
  const Glyph = iconFor(mention.icon || '');
  const state = mention.state || '';
  const StateGlyph = STATE_GLYPHS[state] || null;
  const kind = (locale === 'he' ? mention.kind_he : mention.kind_en) || '';
  const word = stateWord(mention, locale);
  const title = [kind, word].filter(Boolean).join(' · ');
  return (
    <span className={`mention-chip${state ? ` is-${state}` : ''}`} title={title}>
      <span className="mention-chip-glyph" aria-hidden="true"><Glyph size={12} /></span>
      <Name className="mention-chip-name">{mention.current_label || mention.label}</Name>
      {kind ? <span className="mention-chip-kind">{kind}</span> : null}
      {StateGlyph ? (
        <span className="mention-chip-state">
          <StateGlyph size={12} aria-hidden="true" />
          <span className="mention-chip-word">{word}</span>
        </span>
      ) : null}
    </span>
  );
}

// The composer's own strip: what is attached to the sentence right now.
export function AttachedRefs({ locale, refs }) {
  if (!refs || !refs.length) return null;
  return (
    <p className="mention-strip" role="status">
      <span className="mention-strip-label">
        {pageText(locale, 'Pointing at', 'מצביע על')}
      </span>
      {refs.map((ref) => (
        <Chip
          key={`${ref.type}:${ref.id}`}
          locale={locale}
          mention={{ type: ref.type, id: ref.id, label: ref.label, icon: ref.icon, kind_he: ref.kindHe, kind_en: ref.kindEn }}
        />
      ))}
    </p>
  );
}

// The finished turn's strip: what each reference came back as. A row for every
// reference, including the ones that did not come back clean, which is the whole
// difference between this and the product that drops them quietly.
export function ResolvedRefs({ locale, mentions }) {
  if (!mentions || !mentions.length) return null;
  const changed = mentions.filter((m) => m.state === 'changed' && m.current_label && m.current_label !== m.label);
  return (
    <div className="mention-strip mention-strip-resolved">
      <span className="mention-strip-label">
        {pageText(locale, 'This question pointed at', 'השאלה הזו הצביעה על')}
      </span>
      {mentions.map((mention) => (
        <Chip key={`${mention.type}:${mention.id}`} locale={locale} mention={mention} />
      ))}
      {changed.map((mention) => (
        // A name that moved is stated rather than swapped silently: the chip
        // shows what the store holds now, and this says what the question said.
        <span className="mention-chip-was" key={`was:${mention.type}:${mention.id}`}>
          {pageText(locale, 'was', 'היה')} <Name>{mention.label}</Name>
        </span>
      ))}
    </div>
  );
}

// THE HIGHLIGHT BEHIND THE TEXTAREA.
//
// A chip in the sentence itself, without a rich-text editor. A contenteditable
// in a right-to-left dock is a large bidi-fragile build that buys nothing this
// does not: the overlay renders the same string with the same metrics one layer
// down, paints a background behind the characters a reference covers, and the
// textarea's own glyphs sit on top of it. The caret, selection, undo and input
// method all stay the browser's.
//
// IT CARRIES NO GLYPH, and that is a constraint rather than an omission. Every
// character in this layer has to occupy exactly the width the same character
// occupies in the textarea, so anything inserted here that is not in the
// textarea moves the highlight off the words it belongs to. The glyph lives in
// the strip below, where it costs nothing to be right.
//
// Direction comes from Prose, which is the shell's own dir="auto" block: the
// question may be written in either language and the overlay has to resolve the
// same way the textarea does, from the same first strong character.
export const MentionOverlay = React.forwardRef(function MentionOverlay({ text, refs }, ref) {
  const runs = chipRuns(text, refs);
  if (!runs.some((run) => run.chip)) return null;
  return (
    <Prose as="div" ref={ref} className="mention-overlay" aria-hidden="true">
      {runs.map((run, index) => (
        run.chip
          ? <span className="mention-mark" key={`chip-${index}`}>{run.text}</span>
          : <React.Fragment key={`plain-${index}`}>{run.text}</React.Fragment>
      ))}
    </Prose>
  );
});
