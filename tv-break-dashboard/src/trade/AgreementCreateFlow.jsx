import React, { useEffect, useRef, useState } from 'react';
import { Dialog } from '../studio/modal';
import { Button } from '../studio/actions';
import { InputControl, SelectControl } from '../studio/dom-controls';
import { ErrorState, LoadingState, Status } from '../studio';
import { CircleCheck, TriangleAlert, Upload } from 'lucide-react';
import { Code, Figure, Name, Prose } from '../shell/bidi';
import { formatNumber, pageText } from '../shell/format';
import { levelOptions } from './trade-vocabulary';
import {
  createAgreement, loadJob, refusalText, startExtraction, uploadDocument,
} from './trade-api';

// Bringing an agreement in: create the record, attach the signed document, read
// it. Three server acts, one dialog, and the dialog says which of the three it
// is on — because the third one calls a language model over a fifty-clause
// Hebrew contract and takes as long as it takes.
//
// THE PROGRESS IS THE JOB'S OWN. The extraction runs as a background job and
// this dialog polls it. What it reports is the job record: running, done with
// its counts, or failed with the server's own error. There is no simulated bar,
// no percentage invented from elapsed time, and no success shown before the job
// says so. When the provider is unavailable the extraction fails and says that,
// and the agreement it already created stays — a half-finished intake is a state
// the operator can see and finish, not a rollback that loses the upload.

const POLL_MS = 1500;

function Field({ label, hint, children }) {
  return (
    <label className="trd-field">
      <span className="trd-field-label">{label}</span>
      {children}
      {hint ? <span className="trd-field-hint">{hint}</span> : null}
    </label>
  );
}

export default function AgreementCreateFlow({ locale = 'he', notify = () => {}, onClose, onCreated }) {
  const [title, setTitle] = useState('');
  const [level, setLevel] = useState('agency_framework');
  const [kind, setKind] = useState('agency');
  const [party, setParty] = useState('');
  const [from, setFrom] = useState('');
  const [to, setTo] = useState('');
  const [file, setFile] = useState(null);
  // 'form' | 'creating' | 'extracting' | 'done' | 'failed'
  const [phase, setPhase] = useState('form');
  const [error, setError] = useState(null);
  const [created, setCreated] = useState(null);
  const [job, setJob] = useState(null);
  const initialFocus = useRef(null);
  const he = locale === 'he';

  // The poll lives here rather than in the submit handler so a dialog left open
  // keeps following the job, and closing the dialog stops the poll.
  useEffect(() => {
    if (phase !== 'extracting' || !job || !job.job_id) return undefined;
    let alive = true;
    const timer = window.setInterval(async () => {
      try {
        const record = await loadJob(job.job_id);
        if (!alive) return;
        setJob(record);
        if (record.status === 'done') setPhase('done');
        if (record.status === 'failed') { setError(null); setPhase('failed'); }
      } catch (failure) {
        if (!alive) return;
        setError(failure);
        setPhase('failed');
      }
    }, POLL_MS);
    return () => { alive = false; window.clearInterval(timer); };
  }, [phase, job]);

  // A start date is required by the store and it is right that it is: an
  // agreement whose own start nobody knows is a document that has not been read
  // yet. An empty end date is a real answer and means open-ended, which the
  // store records against its own sentinel rather than leaving unmeasurable.
  const canSubmit = title.trim().length > 0 && Boolean(file) && Boolean(from) && phase === 'form';

  async function submit(event) {
    event.preventDefault();
    if (!canSubmit) return;
    setPhase('creating');
    setError(null);
    try {
      const head = await createAgreement({
        title: title.trim(),
        level,
        counterparty: party.trim() ? { kind, name: party.trim() } : null,
        window: { starts_on: from, ends_on: to || null },
      });
      const document = await uploadDocument(head.agreement_id, file);
      setCreated({ agreement: head, document });
      notify(
        `Agreement ${head.agreement_id} created and the document attached.`,
        `ההסכם ${head.agreement_id} נוצר והמסמך צורף.`,
      );
      const started = await startExtraction(head.agreement_id, document.document_id);
      setJob({ job_id: started.job_id, status: 'running' });
      setPhase('extracting');
    } catch (failure) {
      setError(failure);
      setPhase('failed');
    }
  }

  const result = job && job.result ? job.result : null;

  const footer = phase === 'form' ? (
    <>
      <Button type="button" variant="outlined" onClick={onClose}>
        {pageText(locale, 'Cancel', 'ביטול')}
      </Button>
      <Button type="submit" form="trd-create-form" variant="contained" disabled={!canSubmit}>
        <Upload size={14} aria-hidden="true" />
        {pageText(locale, 'Create and read the document', 'יצירה וקריאת המסמך')}
      </Button>
    </>
  ) : (
    <>
      <Button type="button" variant="outlined" onClick={onClose}>
        {pageText(locale, 'Close', 'סגירה')}
      </Button>
      {created ? (
        <Button
          type="button"
          variant="contained"
          onClick={() => onCreated(created.agreement.agreement_id, phase === 'done' ? 'review' : 'detail')}
        >
          {phase === 'done'
            ? pageText(locale, 'Open the review', 'פתיחת הסקירה')
            : pageText(locale, 'Open the agreement', 'פתיחת ההסכם')}
        </Button>
      ) : null}
    </>
  );

  return (
    <Dialog
      open
      onClose={onClose}
      size="wide"
      title={pageText(locale, 'Bring in an agreement', 'הכנסת הסכם')}
      description={pageText(
        locale,
        'The record is created, the signed document is attached, and the document is read into proposed terms. Nothing binds until a reviewer approves it.',
        'הרשומה נוצרת, המסמך החתום מצורף, והמסמך נקרא למונחים מוצעים. דבר אינו מחייב עד שסוקר יאשר.',
      )}
      closeLabel={pageText(locale, 'Close', 'סגירה')}
      initialFocusRef={initialFocus}
      footer={footer}
      className="trd-create-dialog"
    >
      {phase === 'form' ? (
        <form id="trd-create-form" className="trd-form" onSubmit={submit}>
          <Field label={pageText(locale, 'Agreement title', 'שם ההסכם')}>
            <InputControl
              ref={initialFocus}
              type="text"
              value={title}
              required
              onChange={(event) => setTitle(event.target.value)}
              placeholder={he ? 'הסכם מסגרת שנתי — שם הסוכנות' : 'Annual framework — agency name'}
            />
          </Field>

          <Field label={pageText(locale, 'Agreement level', 'רמת ההסכם')}>
            <SelectControl value={level} onChange={(event) => setLevel(event.target.value)}>
              {levelOptions(locale).map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </SelectControl>
          </Field>

          <Field label={pageText(locale, 'Counterparty type', 'סוג הצד להסכם')}>
            <SelectControl value={kind} onChange={(event) => setKind(event.target.value)}>
              <option value="agency">{pageText(locale, 'Agency', 'סוכנות')}</option>
              <option value="advertiser">{pageText(locale, 'Advertiser', 'מפרסם')}</option>
            </SelectControl>
          </Field>

          <Field label={pageText(locale, 'Counterparty name', 'שם הצד להסכם')}>
            <InputControl
              type="text"
              value={party}
              onChange={(event) => setParty(event.target.value)}
            />
          </Field>

          <Field
            label={pageText(locale, 'Effective from', 'בתוקף מיום')}
            hint={pageText(
              locale,
              'Required. Without a start date no commitment inside the agreement has a measurement window.',
              'שדה חובה. בלי מועד תחילה אין לאף התחייבות בהסכם חלון מדידה.',
            )}
          >
            <InputControl
              type="date"
              value={from}
              required
              onChange={(event) => setFrom(event.target.value)}
            />
          </Field>

          <Field
            label={pageText(locale, 'Effective until', 'בתוקף עד יום')}
            hint={pageText(
              locale,
              'Leave it empty for an agreement that runs until one side cancels. It is recorded as open-ended, not as a missing date.',
              'השאירו ריק להסכם שנמשך עד שאחד הצדדים יבטל. הוא יירשם כהסכם ללא מועד סיום, לא כתאריך חסר.',
            )}
          >
            <InputControl type="date" value={to} onChange={(event) => setTo(event.target.value)} />
          </Field>

          <Field
            label={pageText(locale, 'The signed document', 'המסמך החתום')}
            hint={pageText(
              locale,
              'A PDF. A scanned page is read too; the route it took is recorded on the document.',
              'קובץ PDF. גם עמוד סרוק נקרא; נתיב הקריאה נרשם על המסמך.',
            )}
          >
            <InputControl
              type="file"
              accept="application/pdf"
              onChange={(event) => setFile(event.target.files && event.target.files[0] ? event.target.files[0] : null)}
            />
          </Field>

          {file ? (
            <p className="trd-form-note" role="status">
              <Name>{file.name}</Name>
              {' · '}
              <Figure>{formatNumber(Math.round(file.size / 1024), locale)}</Figure>
              {' '}
              {pageText(locale, 'KB', 'ק״ב')}
            </p>
          ) : null}
        </form>
      ) : null}

      {phase === 'creating' ? (
        <LoadingState
          title={pageText(locale, 'Creating the record and attaching the document', 'יוצר את הרשומה ומצרף את המסמך')}
          description={pageText(
            locale,
            'The document is stored with its own checksum, so the version approved later names the exact bytes that were read.',
            'המסמך נשמר עם טביעת אצבע משלו, כך שהגרסה שתאושר בהמשך תנקוב בבייטים המדויקים שנקראו.',
          )}
        />
      ) : null}

      {phase === 'extracting' ? (
        <LoadingState
          title={pageText(locale, 'Reading the document', 'קורא את המסמך')}
          description={pageText(
            locale,
            'Every clause is segmented and matched against the term catalogue. This runs as a background job and is not finished until the job says so.',
            'כל סעיף מפוצל ומותאם לקטלוג המונחים. העבודה מתבצעת ברקע ואינה מסתיימת עד שהעבודה מדווחת על סיום.',
          )}
        >
          <p className="trd-job-line">
            <Status status="info">{pageText(locale, 'Job running', 'עבודה מתבצעת')}</Status>
            {job && job.job_id ? <Code>{job.job_id}</Code> : null}
            {job && job.progress ? (
              <Figure>
                {`${formatNumber(job.progress.done, locale)}/${formatNumber(job.progress.total, locale)}`}
              </Figure>
            ) : (
              <span className="trd-field-hint">
                {pageText(locale, 'The job reports no step count, so none is shown.', 'העבודה אינה מדווחת על מספר שלבים, ולכן לא מוצג מספר.')}
              </span>
            )}
          </p>
        </LoadingState>
      ) : null}

      {phase === 'done' ? (
        <div className="trd-done" role="status">
          <Status status="positive" icon={<CircleCheck size={16} aria-hidden="true" />}>
            {pageText(locale, 'The document was read', 'המסמך נקרא')}
          </Status>
          {result ? (
            <dl className="trd-kv">
              <dt>{pageText(locale, 'Clauses found', 'סעיפים שנמצאו')}</dt>
              <dd><Figure>{formatNumber(result.clauses, locale)}</Figure></dd>
              <dt>{pageText(locale, 'Mapped to a term', 'מופו למונח')}</dt>
              <dd><Figure>{formatNumber(result.mapped, locale)}</Figure></dd>
              <dt>{pageText(locale, 'Mapped to nothing', 'לא מופו לדבר')}</dt>
              <dd><Figure>{formatNumber(result.unmapped, locale)}</Figure></dd>
              <dt>{pageText(locale, 'Proposed terms', 'מונחים מוצעים')}</dt>
              <dd><Figure>{formatNumber(result.instances, locale)}</Figure></dd>
              <dt>{pageText(locale, 'Conflicts detected', 'סתירות שזוהו')}</dt>
              <dd><Figure>{formatNumber(result.conflicts, locale)}</Figure></dd>
            </dl>
          ) : null}
          <p className="trd-field-hint">
            {pageText(
              locale,
              'These are proposals. Nothing changes pricing or placement until each one is reviewed and the agreement is approved.',
              'אלה הצעות. דבר אינו משנה תמחור או שיבוץ עד שכל אחת מהן תיסקר וההסכם יאושר.',
            )}
          </p>
        </div>
      ) : null}

      {phase === 'failed' ? (
        <ErrorState
          title={created
            ? pageText(locale, 'The document was attached but could not be read', 'המסמך צורף אך לא ניתן היה לקרוא אותו')
            : pageText(locale, 'The agreement could not be created', 'לא ניתן היה ליצור את ההסכם')}
          description={created
            ? pageText(
              locale,
              'The record and the document are saved. The reading can be run again from the agreement itself.',
              'הרשומה והמסמך נשמרו. אפשר להריץ את הקריאה שוב מתוך ההסכם עצמו.',
            )
            : pageText(locale, 'Nothing was saved.', 'דבר לא נשמר.')}
        >
          <Prose className="trd-error-detail">
            <TriangleAlert size={14} aria-hidden="true" />
            {(job && job.error) || refusalText(error, locale)}
          </Prose>
        </ErrorState>
      ) : null}
    </Dialog>
  );
}
