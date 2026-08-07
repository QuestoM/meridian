import React from 'react';
import { Figure, Name } from '../shell/bidi';
import { Link2 } from 'lucide-react';
import {
  linkBasisNote,
  linkEmptyNote,
  linkSourceLabel,
  linksWord,
  pageText,
} from './agencies-helpers';

// The advertisers one agency books, each with the provenance of its link:
// observed in the daily spot file, or linked by hand by a planner.
//
// This section lives in its own file because it is rendered on its own by the
// test that measures it, without the drawer's dialogs and inputs around it.
//
// Three states, never mixed: the links loaded, the request failed, or the set
// is genuinely empty and says which file was read to reach that conclusion.
// The names are advertiser trade names in Hebrew, so the name span follows the
// text rather than being forced left to right.
export function LinkedAdvertisers({ state, locale }) {
  const ready = state.status === 'ready';
  const count = state.links.length;
  return (
    <section className="amz-drawer-section">
      <div className="agz-link-head">
        <h3>{pageText(locale, 'Linked advertisers', 'מפרסמים מקושרים')}</h3>
        {ready && count > 0 && (
          <span className="agz-link-count">
            <Figure className="numeric">{count}</Figure>
            <small>{linksWord(count, locale)}</small>
          </span>
        )}
      </div>
      {state.status === 'loading' && (
        <p className="agz-subnote">{pageText(locale, 'Loading links...', 'טוען קישורים...')}</p>
      )}
      {state.status === 'error' && (
        <p className="agz-inline-warn" role="note">{pageText(locale, 'Advertiser links could not be loaded. This is a load failure, not an empty list.', 'קישורי המפרסמים לא נטענו. זהו כשל טעינה, לא רשימה ריקה.')}</p>
      )}
      {ready && count === 0 && (
        <p className="agz-subnote">{linkEmptyNote(state.sourceFile, locale)}</p>
      )}
      {ready && count > 0 && (
        <>
          <ul className="agz-link-list">
            {state.links.map((link) => (
              <li key={link.advertiser} className="agz-link-row">
                <Link2 size={13} aria-hidden="true" />
                <Name className="agz-link-name">{link.advertiser}</Name>
                <span className={`agz-status-chip ${link.source === 'manual' ? 'blue' : 'teal'}`}>{linkSourceLabel(link.source, locale)}</span>
              </li>
            ))}
          </ul>
          {state.sourceFile && (
            <p className="agz-subnote">{linkBasisNote(state.sourceFile, locale)}</p>
          )}
        </>
      )}
    </section>
  );
}

export default LinkedAdvertisers;
