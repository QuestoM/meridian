import React, { useEffect, useMemo, useState } from 'react';
import { Card, CardBody, ErrorState, LoadingState, Status } from '../studio';
import { Button } from '../studio/actions';
import { InputControl, Pressable } from '../studio/dom-controls';
import { Check, Eye, FileText, ListChecks, Search } from 'lucide-react';
import { Code, Name, Prose } from '../shell/bidi';
import { formatNumber, pageText } from '../shell/format';
import { dispositionLabel, dispositionTone } from './trade-vocabulary';
import { irrelevantClassName } from './trade-terms';
import { fetchDocumentBlob, refusalText } from './trade-api';

// The source document, beside the machine's reading of it.
//
// WHY THE PDF IS FETCHED AS BYTES. Pointing an iframe at the API route makes the
// browser perform its own credentialless request, and a 401 or a 404 then renders
// as Chrome's error page inside the review — a failure this surface cannot name
// or recover from. Fetching the blob puts the request back under the app's own
// error handling: either an object URL exists, or the read failed and the pane
// says so with the server's reason.
//
// THE CLAUSE LIST IS THE WORKING VIEW and the page is one click away, because
// reading fifty clauses in a PDF viewport is not reviewing. Selecting a clause
// moves the viewer to that clause's own page, so the reviewer can check the
// machine's quote against the signed page without hunting for it.
//
// SEEN IS AN ACT, NOT A DEFAULT. A clause becomes "read" when the reviewer opens
// it or explicitly marks the rows on screen. Nothing marks a clause read because
// it was rendered somewhere below the fold.

const CLAUSE_WINDOW = 25;

function ClauseRow({ clause, locale, selected, onSelect }) {
  const cited = clause.instanceIds.length;
  return (
    <Pressable
      className="trd-clause"
      data-selected={selected ? 'true' : undefined}
      data-disposition={clause.disposition}
      aria-pressed={selected}
      onClick={() => onSelect(clause)}
    >
      <span className="trd-clause-head">
        <Code className="trd-clause-id">{clause.clause_id}</Code>
        {clause.seen ? (
          <span className="trd-clause-seen" title={pageText(locale, 'Read', 'נקרא')}>
            <Check size={13} aria-hidden="true" />
            <span>{pageText(locale, 'Read', 'נקרא')}</span>
          </span>
        ) : (
          <span className="trd-clause-unseen">{pageText(locale, 'Not read yet', 'לא נקרא')}</span>
        )}
        <Status status={dispositionTone(clause.disposition)}>
          {dispositionLabel(clause.disposition, locale)}
        </Status>
      </span>
      {clause.heading ? <Name className="trd-clause-heading">{clause.heading}</Name> : null}
      <Prose as="span" className="trd-clause-text">{clause.text}</Prose>
      <span className="trd-clause-foot">
        {clause.pages.length > 0 ? (
          <Code className="trd-clause-page">
            {pageText(locale, `page ${clause.pages[0]}`, `עמ׳ ${clause.pages[0]}`)}
          </Code>
        ) : null}
        {clause.isTable ? (
          <span className="trd-chip-quiet">{pageText(locale, 'Table', 'טבלה')}</span>
        ) : null}
        {cited > 0 ? (
          <span className="trd-chip-quiet">
            {pageText(
              locale,
              `${formatNumber(cited, locale)} terms`,
              `${formatNumber(cited, locale)} מונחים`,
            )}
          </span>
        ) : null}
        {clause.irrelevantClass ? (
          <span className="trd-chip-quiet">{irrelevantClassName(clause.irrelevantClass, locale)}</span>
        ) : null}
        {clause.acknowledged ? (
          <Status status="info">{pageText(locale, 'Acknowledged', 'אושר ידנית')}</Status>
        ) : null}
      </span>
      {clause.dispositionReason ? (
        <Prose as="span" className="trd-clause-reason">{clause.dispositionReason}</Prose>
      ) : null}
    </Pressable>
  );
}

// The signed page itself. The viewer is the browser's own; nothing here parses a
// PDF, and the pane says plainly when the read failed rather than leaving an
// empty frame.
function DocumentPage({ agreementId, documentId, page, locale }) {
  const [url, setUrl] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    let created = '';
    setError(null);
    setUrl('');
    fetchDocumentBlob(agreementId, documentId).then(
      (blob) => {
        if (!alive) return;
        created = URL.createObjectURL(blob);
        setUrl(created);
      },
      (failure) => { if (alive) setError(failure); },
    );
    return () => {
      alive = false;
      if (created) URL.revokeObjectURL(created);
    };
  }, [agreementId, documentId]);

  if (error) {
    return (
      <ErrorState
        title={pageText(locale, 'The document could not be read', 'לא ניתן היה לקרוא את המסמך')}
        description={refusalText(error, locale)}
      />
    );
  }
  if (!url) {
    return (
      <LoadingState
        title={pageText(locale, 'Fetching the signed document', 'מביא את המסמך החתום')}
        description={pageText(
          locale,
          'The bytes are fetched through the application, so a refusal is reported here rather than by the browser.',
          'הבייטים מובאים דרך היישום, כך שסירוב מדווח כאן ולא על ידי הדפדפן.',
        )}
      />
    );
  }
  return (
    <iframe
      className="trd-pdf"
      src={page ? `${url}#page=${page}` : url}
      title={pageText(locale, 'The signed agreement document', 'מסמך ההסכם החתום')}
    />
  );
}

export default function ReviewDocumentPane({
  agreementId, documentId, clauses, locale, selected, onSelect, onMarkSeen, canEdit, marking,
}) {
  const [tab, setTab] = useState('clauses');
  const [query, setQuery] = useState('');
  const [visibleCount, setVisibleCount] = useState(CLAUSE_WINDOW);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return clauses;
    return clauses.filter((clause) => (
      clause.clause_id.toLowerCase().includes(needle)
      || clause.text.toLowerCase().includes(needle)
      || String(clause.heading || '').toLowerCase().includes(needle)
    ));
  }, [clauses, query]);

  useEffect(() => { setVisibleCount(CLAUSE_WINDOW); }, [query]);

  const windowed = filtered.slice(0, visibleCount);
  const unseenShown = windowed.filter((clause) => !clause.seen).map((clause) => clause.clause_id);
  const selectedClause = clauses.find((clause) => clause.clause_id === selected) || null;
  const page = selectedClause && selectedClause.pages.length > 0 ? selectedClause.pages[0] : 0;

  return (
    <Card className="trd-doc-pane">
      <CardBody>
        <div className="trd-pane-head">
          <h3>{pageText(locale, 'The signed document', 'המסמך החתום')}</h3>
          <div
            className="trd-tabs"
            role="tablist"
            aria-label={pageText(locale, 'Document views', 'תצוגות המסמך')}
          >
            <Button
              type="button"
              role="tab"
              aria-selected={tab === 'clauses'}
              className={tab === 'clauses' ? 'active' : ''}
              onClick={() => setTab('clauses')}
            >
              <ListChecks size={14} aria-hidden="true" />
              {pageText(locale, 'Clauses', 'הסעיפים')}
            </Button>
            <Button
              type="button"
              role="tab"
              aria-selected={tab === 'page'}
              className={tab === 'page' ? 'active' : ''}
              onClick={() => setTab('page')}
            >
              <FileText size={14} aria-hidden="true" />
              {pageText(locale, 'The page', 'העמוד')}
            </Button>
          </div>
        </div>

        {tab === 'clauses' ? (
          <div className="trd-clause-tools">
            <span className="trd-search">
              <Search size={15} aria-hidden="true" />
              <InputControl
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={pageText(locale, 'Search the clauses', 'חיפוש בסעיפים')}
                aria-label={pageText(locale, 'Search the clauses', 'חיפוש בסעיפים')}
              />
            </span>
            {canEdit && unseenShown.length > 0 ? (
              <Button
                type="button"
                variant="outlined"
                disabled={marking}
                onClick={() => onMarkSeen(unseenShown)}
              >
                <Eye size={14} aria-hidden="true" />
                {pageText(
                  locale,
                  `Mark these ${formatNumber(unseenShown.length, locale)} as read`,
                  `סימון ${formatNumber(unseenShown.length, locale)} אלה כנקראו`,
                )}
              </Button>
            ) : null}
          </div>
        ) : null}

        {tab === 'clauses' ? (
          <div className="trd-clause-list">
            {windowed.map((clause) => (
              <ClauseRow
                key={clause.clause_id}
                clause={clause}
                locale={locale}
                selected={clause.clause_id === selected}
                onSelect={onSelect}
              />
            ))}
            {filtered.length === 0 ? (
              <p className="trd-field-hint">
                {pageText(locale, 'No clause matches that search.', 'אין סעיף שתואם את החיפוש.')}
              </p>
            ) : null}
            {windowed.length < filtered.length ? (
              <div className="trd-window-more" role="status">
                <span>
                  {pageText(
                    locale,
                    `Showing ${formatNumber(windowed.length, locale)} of ${formatNumber(filtered.length, locale)} clauses`,
                    `מוצגים ${formatNumber(windowed.length, locale)} מתוך ${formatNumber(filtered.length, locale)} סעיפים`,
                  )}
                </span>
                <Button
                  type="button"
                  variant="outlined"
                  onClick={() => setVisibleCount((count) => count + CLAUSE_WINDOW)}
                >
                  {pageText(locale, 'Show the next clauses', 'הצגת הסעיפים הבאים')}
                </Button>
              </div>
            ) : null}
          </div>
        ) : (
          <DocumentPage
            agreementId={agreementId}
            documentId={documentId}
            page={page}
            locale={locale}
          />
        )}
      </CardBody>
    </Card>
  );
}
