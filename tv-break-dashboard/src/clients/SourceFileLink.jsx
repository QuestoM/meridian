import React from 'react';
import { Button } from '../studio/actions';
import { Code } from '../shell/bidi';
import { pageText } from '../shell/format';

// A source file, named and openable.
//
// Every counted figure on this destination carries the file it was read out of,
// and that name was a dead end: a reader who wanted to know what was in the file,
// when it arrived or whether it is still the one being read had to find the Data
// screen themselves and pick it out of a list. The name IS the way there.
//
// It opens the Data screen's inputs view focused on that file, because that is
// where a daily file's state lives — when it was uploaded, how many rows were
// read, what the checks said, and the control that replaces it. The Source files
// table is a different question (what is on disk) and does not carry the daily
// input at all, which is why this does not go there.
//
// The address follows the shell's routing law: the domain is the hash, and the
// scope travels in the real query string. The Data screen resolves the NAME to
// the input it belongs to, because it is the screen holding the upload status
// and this one is not — a component that guessed the kind from a filename would
// be a second copy of a mapping the server already publishes.

// The shell's own canonical address for that screen, not the `#Data` alias.
// Measured: linking to the alias works, and the shell immediately rewrites the
// address to `?sources=inputs#Sources` — so the alias costs a history entry
// that is corrected a moment later, and anyone who copies the URL mid-rewrite
// gets the wrong one. The view is selected by the shell's `sources` param;
// naming it twice would be two sources of truth for one thing.
export function sourceFileAddress(name) {
  const params = new URLSearchParams(
    typeof window === 'undefined' ? '' : window.location.search);
  params.set('sources', 'inputs');
  params.set('sourceFile', String(name || ''));
  // Nothing on this screen is scoped by these, and carrying them forward would
  // leave a clients-page filter in the address of a different workspace.
  ['clients', 'client', 'agreement', 'campaign', 'source', 'sourceView']
    .forEach((key) => params.delete(key));
  const path = typeof window === 'undefined' ? '' : window.location.pathname;
  return `${path}?${params.toString()}#Sources`;
}

export function openSourceFile(name) {
  if (typeof window === 'undefined') {
    return;
  }
  // Two steps, in this order, and neither is decoration.
  //
  // The scope goes into the address first, without touching the hash, so it is
  // already there when the shell reads it. Then the hash is assigned NATIVELY,
  // which is what actually moves the shell: pushState changes the address and
  // tells nobody, and a synthetic HashChangeEvent is not what the shell listens
  // for. Measured — with pushState alone the address was perfect and the screen
  // never changed. goToView, which works today, assigns the hash the same way.
  const address = sourceFileAddress(name);
  const query = address.slice(address.indexOf('?'), address.indexOf('#'));
  window.history.replaceState(
    { workspace: 'sources', sourceFile: String(name || '') },
    '',
    `${window.location.pathname}${query}${window.location.hash}`,
  );
  if (decodeURIComponent(window.location.hash.replace(/^#/, '')) === 'Sources') {
    // Already there: assigning the same hash fires nothing, so the page is told
    // directly that its scope moved.
    window.dispatchEvent(new PopStateEvent('popstate'));
    return;
  }
  window.location.hash = 'Sources';
}

export default function SourceFileLink({ name, locale }) {
  const shown = String(name || '').trim();
  if (!shown) {
    return null;
  }
  return (
    <Button
      type="button"
      variant="text"
      className="clients-link source-file-link"
      onClick={() => openSourceFile(shown)}
      title={pageText(locale, `Open ${shown} on the Data screen`, `פתחו את ${shown} במסך הנתונים`)}
    >
      <Code>{shown}</Code>
    </Button>
  );
}
