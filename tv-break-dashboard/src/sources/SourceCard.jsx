import React, { useRef, useState } from 'react';
import { Button, Tooltip } from '@mui/material';
import { AlertTriangle, CheckCircle2, FileWarning, Rows3, Upload } from 'lucide-react';
import { Numeric, formatNumber } from '../shell/format';
import {
  CADENCE_LABELS,
  CADENCE_NOTES,
  STATE_LABELS,
  label,
  serverText,
  text,
} from './sources-copy';
import { checkFile, uploadFile } from './sources-api';
import { fieldLabel } from './sources-fields';
import { acceptedVerdict, findingMessage, stateTone, visibleFindings } from './sources-findings';
import SourceChecks from './SourceChecks';

function formatSize(bytes, locale) {
  const number = Number(bytes) || 0;
  if (number <= 0) return '-';
  const kilobytes = number / 1024;
  // A 150-byte file is not a zero-byte file. Rounding it to 0 KB would say a
  // file with content is empty, which is the one thing a size column owes.
  if (kilobytes < 1) return `< 1 KB`;
  if (kilobytes < 1024) return `${formatNumber(Math.round(kilobytes), locale)} KB`;
  return `${formatNumber(Math.round(kilobytes / 1024), locale)} MB`;
}

function formatWhen(value, locale) {
  if (!value) return '-';
  const when = new Date(value);
  if (Number.isNaN(when.getTime())) return String(value);
  return when.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US');
}

// The rows the finding is about. A column and a count leave a steward searching
// a 175-row file by hand, so the row numbers print beside the sentence. Each
// number stands alone rather than in a comma-separated run, because a grouped
// row number (1,240) inside a comma list reads as two rows. When the server
// capped the list, the count it did not print is stated rather than dropped.
function FindingRows({ finding, locale }) {
  const rows = Array.isArray(finding.rows) ? finding.rows : [];
  if (rows.length === 0) return null;
  const total = Number(finding.rows_total) || rows.length;
  const more = total - rows.length;
  return (
    <span className="finding-rows">
      <Tooltip title={text('findingRowsNote', locale)} arrow placement="top">
        <span className="finding-rows-label">{total === 1 ? text('findingRow', locale) : text('findingRows', locale)}</span>
      </Tooltip>
      {rows.map((row) => (
        <span className="finding-row" key={row}>
          <Numeric>{formatNumber(row, locale)}</Numeric>
        </span>
      ))}
      {more > 0 ? (
        <span className="finding-rows-more">
          <Numeric>{`+${formatNumber(more, locale)}`}</Numeric>
          <span>{text('findingRowsMore', locale)}</span>
        </span>
      ) : null}
    </span>
  );
}

// The rows of a refusal. What each one prints, and whether it prints at all, is
// decided in sources-findings.js: the chip is a column name or the word for
// what the finding is about, never the internal token that used to reach the
// screen, and a finding that only restates the sentence printed above it does
// not print it a second time. `printed` is that sentence.
function Findings({ findings, locale, printed }) {
  const lines = visibleFindings(findings, locale, printed);
  if (lines.length === 0) return null;
  return (
    <ul className="source-findings">
      {lines.map(({ finding, chip, message }, index) => (
        <li key={`${finding.code}-${index}`} className={`${finding.severity === 'error' ? 'finding bad' : 'finding warn'}${chip ? '' : ' no-chip'}`}>
          {chip ? <span className="finding-column" dir={chip.dir}>{chip.text}</span> : null}
          <span className="finding-detail">
            {/* The server's own words, quoted verbatim. dir auto so a sentence
                that stayed English renders as one left-to-right run inside a
                Hebrew card instead of being reordered around its punctuation. */}
            {message ? <span className="finding-message" dir="auto">{message}</span> : null}
            <FindingRows finding={finding} locale={locale} />
          </span>
        </li>
      ))}
    </ul>
  );
}

// The other files of this kind on disk that the engine does not read. Without
// this list the file an operator just sent could sit on disk named on no
// screen, while the card reported the file the engine reads as in use with
// nothing to do. Each one carries the server's own reason, and the one that
// arrived after the file being read is the one that is a problem rather than
// an archive, so it is the one that is marked.
function StoredFiles({ input, locale, onOpenFile }) {
  const stored = Array.isArray(input.stored_unread) ? input.stored_unread : [];
  if (stored.length === 0) return null;
  const total = Number(input.stored_unread_total) || stored.length;
  const more = total - stored.length;
  return (
    <div className="source-stored">
      <p className="source-stored-head">{text('storedUnread', locale)}</p>
      <ul>
        {stored.map((file) => (
          <li key={file.path} className={file.arrived_after_live ? 'warn' : undefined}>
            <span className="source-stored-line">
              <Button className="link-figure" type="button" onClick={() => onOpenFile(file.path)}>
                <FileWarning size={12} />
                <span dir="ltr">{file.filename}</span>
              </Button>
              {file.rows === null || file.rows === undefined ? null : (
                <span className="source-stored-rows">
                  <Numeric>{formatNumber(file.rows, locale)}</Numeric>
                  <span>{text('rows', locale)}</span>
                </span>
              )}
              <Numeric>{formatWhen(file.last_modified, locale)}</Numeric>
            </span>
            <span className="source-stored-why">{serverText(file.reason, locale)}</span>
          </li>
        ))}
      </ul>
      {more > 0 ? (
        <p className="source-stored-more">
          <Numeric>{`+${formatNumber(more, locale)}`}</Numeric>
          <span>{text('storedUnreadMore', locale)}</span>
        </p>
      ) : null}
    </div>
  );
}

// One fact on the card. Which of them print is the reader's choice, so each is
// resolved by key rather than written into the layout. A row count is the one
// fact that opens something: the rows behind it.
function fieldValue(key, input, locale, onOpenRows) {
  if (key === 'rows') {
    if (!input.exists) return <span className="source-none">-</span>;
    return (
      <Button className="link-figure" type="button" onClick={() => onOpenRows(input)} title={text('showRows', locale)}>
        <Rows3 size={12} />
        <Numeric>{formatNumber(input.rows, locale)}</Numeric>
      </Button>
    );
  }
  if (key === 'columns') {
    // The server withholds the name of any column naming a channel this
    // operator does not own, and sends how many it withheld, so the two add
    // back up to the number of columns the file really has. A count that
    // dropped by three because three names were withheld would be a false one.
    const columns = (input.columns || []).length + (Number(input.columns_withheld) || 0);
    return <Numeric>{input.exists ? formatNumber(columns, locale) : '-'}</Numeric>;
  }
  if (key === 'size') {
    return <Numeric>{input.exists ? formatSize(input.size_bytes, locale) : '-'}</Numeric>;
  }
  if (key === 'updated') {
    return <Numeric>{formatWhen(input.last_modified, locale)}</Numeric>;
  }
  if (key === 'path') {
    return <span className="source-card-file" dir="ltr">{input.path}</span>;
  }
  if (key === 'cadence') {
    return <span>{label(CADENCE_LABELS, input.cadence, locale)}</span>;
  }
  if (key === 'lastChecked') {
    const checked = input.last_validation && input.last_validation.checked_at;
    if (!checked) return <span className="source-none">-</span>;
    return <Numeric>{formatWhen(checked, locale)}</Numeric>;
  }
  return null;
}

// One input, and everything true about it: what state it is in, what the
// engine reads for it, what an upload would actually do, and the file's own
// facts. Nothing here is a badge without a consequence beside it.
export function SourceCard({ input, locale, canEdit, canEditReason, fields, onOpenRows, onOpenFile, onChanged, notify }) {
  const fileRef = useRef(null);
  const [busy, setBusy] = useState('');
  const [candidate, setCandidate] = useState(null);
  const [check, setCheck] = useState(null);

  const name = locale === 'he' ? input.label_he || input.label_en : input.label_en;
  const state = String(input.state || 'missing');
  // The state's own tone, unless the server's remedy says this live file's own
  // last check came back with a warning. The rule is in sources-findings.js, run
  // there by a test exactly as the card runs it.
  const tone = stateTone(input);
  const lastValidation = input.last_validation || null;

  async function handleChosen(event) {
    const file = event.target.files && event.target.files[0];
    if (fileRef.current) fileRef.current.value = '';
    if (!file) return;
    setBusy('check');
    setCheck(null);
    setCandidate(file);
    const result = await checkFile(input.kind, file);
    setBusy('');
    setCheck(result);
    if (!result.ok || !result.accepted) {
      notify(
        `${input.label_en}: the file was refused and nothing was replaced.`,
        `${input.label_he}: הקובץ נדחה ושום דבר לא הוחלף.`,
      );
    }
  }

  async function handleCommit() {
    if (!candidate) return;
    setBusy('upload');
    const result = await uploadFile(input.kind, candidate);
    setBusy('');
    if (!result.ok) {
      setCheck({ ok: false, accepted: false, detail: result.detail, detail_he: result.detail_he, errors: result.errors, findings: result.findings });
      return;
    }
    setCheck(null);
    setCandidate(null);
    const rows = formatNumber(result.rows, locale);
    const consequence = serverText(result.consequence, locale);
    notify(
      `${input.label_en}: ${result.rows} rows uploaded. ${serverText(result.consequence, 'en')}`,
      `${input.label_he}: הועלו ${rows} שורות. ${consequence}`,
    );
    if (onChanged) await onChanged();
  }

  const refused = check && (!check.ok || !check.accepted);
  const accepted = check && check.ok && check.accepted;
  // The refusal's own sentence, resolved once. It is printed as the reason and
  // then handed to the findings, which is how they know not to print it again:
  // a refusal made of one finding carries that finding's sentence here too, and
  // the two adjacent copies of it were what a reader saw.
  const refusalDetail = check && check.detail ? findingMessage({ message: check.detail, message_he: check.detail_he }, locale) : '';
  // A file that passes every check and that the engine will not read is not
  // good news, and a green tick over it is how a steward commits a file whose
  // airing date cannot win and walks away believing the plan moved. The same is
  // true of a file the engine WILL read that carries no rows, and of one whose
  // rows carry something the engine cannot read: all three were printed teal
  // under "the file passed every check" over an enabled commit button, and the
  // third one emptied nothing and lost every daypart instead. The whole rule is
  // in sources-findings.js, run there by a test exactly as the card runs it.
  const verdict = accepted ? acceptedVerdict(check) : null;
  const warned = Boolean(verdict && verdict.tone === 'warn');

  return (
    <article className={`source-card tone-${tone}`} data-kind={input.kind} data-state={state}>
      <header className="source-card-head">
        <div className="source-card-title">
          <strong>{name}</strong>
          <span className="source-card-file" dir="ltr">{input.filename}</span>
        </div>
        <span className={`source-state ${tone}`}>{label(STATE_LABELS, state, locale)}</span>
      </header>

      <dl className="source-facts">
        {(fields || []).map((key) => (
          <div key={key}>
            <dt>{fieldLabel(key, locale)}</dt>
            <dd>{fieldValue(key, input, locale, onOpenRows)}</dd>
          </div>
        ))}
      </dl>
      <p className="source-facts-note">
        <span>{text('fromTheFile', locale)}</span>
        <Tooltip title={label(CADENCE_NOTES, input.cadence, locale)} arrow placement="top">
          <span className="source-cadence">{label(CADENCE_LABELS, input.cadence, locale)}</span>
        </Tooltip>
      </p>

      <div className="source-reads">
        <span>{text('engineReads', locale)}</span>
        {input.engine_reads ? (
          <Button className="link-figure" type="button" onClick={() => onOpenFile(input.engine_reads)}>
            <span dir="ltr">{input.engine_reads}</span>
          </Button>
        ) : (
          <span className="source-none">{text('engineReadsNone', locale)}</span>
        )}
      </div>
      <StoredFiles input={input} locale={locale} onOpenFile={onOpenFile} />
      {/* What to do, and what an upload here does, are two answers, and each
          one says which question it is answering. Measured before the labels:
          the two sentences sat adjacent and unlabelled on a shadowed card and
          read as one paragraph that contradicted itself.

          Both are about this KIND as it stands, and both stand down the moment
          a real file is on the table: the verdict below answers for that file.
          Measured before the pair moved together: the remedy printed "nothing
          to do" directly over an amber verdict saying the file in hand carries
          two warnings, which is the same defect the consequence had already
          been taken out of. */}
      {check ? null : (
        <>
          <p className={`source-remedy ${tone}`}>
            <span className="source-line-label">{text('remedyField', locale)}</span>
            <span>{serverText(input.remedy, locale)}</span>
          </p>
          <p className="source-consequence">
            <span className="source-line-label">{text('consequenceField', locale)}</span>
            <span>{serverText(input.consequence, locale)}</span>
          </p>
        </>
      )}
      <SourceChecks checks={input.checks} locale={locale} />

      {lastValidation && !check ? (
        <div className="source-last-check">
          <span>{text('lastChecked', locale)}</span>
          <Numeric>{formatWhen(lastValidation.checked_at, locale)}</Numeric>
          {/* The name of the file that was checked, beside the moment it was
              checked. A timestamp with no filename beside it is how a check of
              one file reads as a check of the one the card names. */}
          {lastValidation.filename ? (
            <span className="source-card-file" dir="ltr">{lastValidation.filename}</span>
          ) : null}
          <Findings findings={lastValidation.findings} locale={locale} />
        </div>
      ) : null}

      {refused ? (
        <div className="source-verdict bad" role="alert">
          <FileWarning size={14} />
          <div>
            <strong>{text('refused', locale)}</strong>
            {refusalDetail ? <p className="source-verdict-detail" dir="auto">{refusalDetail}</p> : null}
            <Findings findings={check.findings && check.findings.length ? check.findings : (check.errors || []).map((message) => ({ column: '', code: 'error', message, severity: 'error' }))} locale={locale} printed={refusalDetail} />
          </div>
        </div>
      ) : null}

      {accepted ? (
        <div className={`source-verdict ${verdict.tone}`} role="status">
          {warned ? <AlertTriangle size={14} /> : <CheckCircle2 size={14} />}
          <div>
            <strong>{text(verdict.heading, locale)}</strong>
            <p className="source-verdict-detail">
              <Numeric>{formatNumber(check.rows, locale)}</Numeric> {text('rows', locale)}
            </p>
            {check.saves_to ? (
              <p className="source-verdict-detail">
                <span>{text('savesTo', locale)}</span>
                <span className="source-card-file" dir="ltr">{check.saves_to}</span>
              </p>
            ) : null}
            <p className="source-verdict-detail">{serverText(check.consequence, locale)}</p>
            <Findings findings={check.findings} locale={locale} />
          </div>
        </div>
      ) : null}

      <input ref={fileRef} type="file" accept=".csv" hidden onChange={handleChosen} />

      <div className="source-actions">
        {accepted ? (
          <>
            <Tooltip title={canEdit ? '' : canEditReason || ''} arrow placement="top">
              <span>
                <Button className="primary-button compact" type="button" variant="contained" disabled={!canEdit || busy === 'upload'} onClick={handleCommit}>
                  <Upload size={14} />
                  {busy === 'upload' ? text('uploading', locale) : text('commit', locale)}
                </Button>
              </span>
            </Tooltip>
            <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => fileRef.current && fileRef.current.click()}>
              {text('discard', locale)}
            </Button>
          </>
        ) : (
          <Tooltip title={canEdit ? '' : canEditReason || ''} arrow placement="top">
            <span>
              <Button className="secondary-button compact" type="button" variant="outlined" disabled={!canEdit || busy === 'check'} onClick={() => fileRef.current && fileRef.current.click()}>
                {busy === 'check' ? <AlertTriangle size={14} /> : <Upload size={14} />}
                {busy === 'check' ? text('checking', locale) : text('chooseFile', locale)}
              </Button>
            </span>
          </Tooltip>
        )}
      </div>
    </article>
  );
}

export default SourceCard;
