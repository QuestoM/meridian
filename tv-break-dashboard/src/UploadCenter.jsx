import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '@mui/material';
import { AlertTriangle, CheckCircle2, Database, RefreshCcw, Upload } from 'lucide-react';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || 'http://127.0.0.1:8000';

function pageText(locale, en, he) {
  return locale === 'he' ? he : en;
}

function normalizeRows(value) {
  return Array.isArray(value) ? value : [];
}

const cadenceMeta = {
  weekly: {
    en: 'Weekly (start of week)',
    he: 'שבועי (תחילת שבוע)',
    noteEn: 'The channel programme lineup, refreshed at the start of each week.',
    noteHe: 'לוח התוכניות של הערוץ, מתעדכן בתחילת כל שבוע.',
  },
  daily: {
    en: 'Daily (start of day)',
    he: 'יומי (תחילת יום)',
    noteEn: "The day's booked ads (Wally), loaded each morning for the next broadcast day.",
    noteHe: 'הפרסומות שהוזמנו ליום (Wally), נטענות בכל בוקר ליום השידור הבא.',
  },
  reference: {
    en: 'Reference data (periodic refresh)',
    he: 'נתוני רפרנס (רענון תקופתי)',
    noteEn: 'Historical ratings the model is built on. Refreshed occasionally, not every day.',
    noteHe: 'נתוני רייטינג היסטוריים שעליהם בנוי המודל. מתרעננים מדי פעם, לא מדי יום.',
  },
  config: {
    en: 'Configuration',
    he: 'תצורה',
    noteEn: 'Not channel data: advertiser terms (also editable in the Advertisers screen) and the pricing rate card.',
    noteHe: 'לא נתוני ערוץ: תנאי מפרסמים (ניתנים לעריכה גם במסך "מפרסמים") וכרטיס התעריפים.',
  },
};

const cadenceOrder = ['weekly', 'daily', 'reference', 'config'];

function formatTimestamp(value, locale) {
  if (!value) {
    return pageText(locale, 'No data yet', 'אין נתונים עדיין');
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US');
}

function InputCard({ input, locale, onUploaded, notify }) {
  const fileRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [errors, setErrors] = useState([]);

  const label = pageText(locale, input.label_en || input.kind, input.label_he || input.label_en || input.kind);
  const valid = Boolean(input.valid) && Boolean(input.exists);
  // Law 9 honesty: a stored upload can be shadowed by a reference file, so the
  // optimizer is not using it. in_use === false means stored-but-not-in-use; a
  // missing in_use (older payloads) keeps the existing truthful display.
  const storedNotInUse = valid && input.in_use === false;
  const inUseReason = input.in_use_reason || '';
  const warnings = normalizeRows(input.warnings);

  async function handleFile(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) {
      return;
    }
    setUploading(true);
    setErrors([]);
    const body = new FormData();
    body.append('file', file);
    try {
      const response = await fetch(`${API_BASE}/api/uploads/${input.kind}`, {
        method: 'POST',
        body,
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const reported = normalizeRows(payload?.errors);
        const detail = payload?.detail || pageText(locale, 'Upload rejected', 'ההעלאה נדחתה');
        setErrors(reported.length ? reported : [detail]);
        notify(
          `${label}: ${detail}`,
          `${label}: ${detail}`,
        );
      } else {
        const rows = Number(payload?.rows || 0);
        const uploadWarnings = normalizeRows(payload?.warnings);
        const warningSuffix = uploadWarnings.length
          ? pageText(locale, ` with ${uploadWarnings.length} warning(s)`, ` עם ${uploadWarnings.length} אזהרות`)
          : '';
        // Honest signal: do not imply the optimizer is using the file when a
        // reference file shadows it. Append the stored-not-in-use note instead
        // of a clean success line.
        const notInUseSuffix = payload?.in_use === false
          ? pageText(
              locale,
              ` Stored, not in use: ${payload?.in_use_reason || 'a reference file currently takes priority.'}`,
              ` נשמר, לא בשימוש: ${payload?.in_use_reason || 'קובץ רפרנס מקבל כרגע עדיפות.'}`,
            )
          : '';
        notify(
          `${label}: uploaded ${rows.toLocaleString('en-US')} rows${warningSuffix}.${notInUseSuffix}`,
          `${label}: הועלו ${rows.toLocaleString('he-IL')} שורות${warningSuffix}.${notInUseSuffix}`,
        );
        await onUploaded();
      }
    } catch (error) {
      setErrors([error.message]);
      notify(
        `${label}: upload failed (${error.message}).`,
        `${label}: ההעלאה נכשלה (${error.message}).`,
      );
    } finally {
      setUploading(false);
      if (fileRef.current) {
        fileRef.current.value = '';
      }
    }
  }

  return (
    <div className="upload-card">
      <div className="upload-card-head">
        <div>
          <strong>{label}</strong>
          <span className="upload-filename" dir="ltr">{input.filename}</span>
        </div>
        {storedNotInUse ? (
          <span className="upload-badge stored" style={{ color: '#b7791f', borderColor: '#b7791f' }}>
            <AlertTriangle size={13} />
            {pageText(locale, 'Stored, not in use', 'נשמר, לא בשימוש')}
          </span>
        ) : (
          <span className={valid ? 'upload-badge valid' : 'upload-badge invalid'}>
            {valid ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
            {valid ? pageText(locale, 'Valid', 'תקין') : pageText(locale, 'Needs review', 'דורש בדיקה')}
          </span>
        )}
      </div>
      <div className="upload-card-meta">
        <div>
          <span>{pageText(locale, 'Rows', 'שורות')}</span>
          <strong>{input.exists ? Number(input.rows || 0).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US') : '—'}</strong>
        </div>
        <div>
          <span>{pageText(locale, 'Last updated', 'עודכן לאחרונה')}</span>
          <strong>{formatTimestamp(input.last_modified, locale)}</strong>
        </div>
      </div>
      {!input.exists && (
        <p className="upload-empty">{pageText(locale, 'No file uploaded yet for this input.', 'עדיין לא הועלה קובץ עבור קלט זה.')}</p>
      )}
      {storedNotInUse && (
        <p className="upload-empty" style={{ color: '#b7791f' }}>
          {inUseReason || pageText(locale, 'Stored but not used by the optimizer: a reference file currently takes priority.', 'נשמר אך לא בשימוש האופטימייזר: קובץ רפרנס מקבל כרגע עדיפות.')}
        </p>
      )}
      {warnings.length > 0 && (
        <ul className="upload-warnings">
          {warnings.map((warning, index) => (
            <li key={index}>{warning}</li>
          ))}
        </ul>
      )}
      {errors.length > 0 && (
        <ul className="upload-errors">
          {errors.map((error, index) => (
            <li key={index}>{error}</li>
          ))}
        </ul>
      )}
      <input
        ref={fileRef}
        type="file"
        accept=".csv"
        hidden
        onChange={handleFile}
      />
      <Button
        className="secondary-button compact"
        type="button"
        variant="outlined"
        disabled={uploading}
        onClick={() => fileRef.current && fileRef.current.click()}
      >
        {uploading ? <RefreshCcw size={14} className="upload-spinner" /> : <Upload size={14} />}
        {uploading ? pageText(locale, 'Uploading...', 'מעלה...') : pageText(locale, 'Upload CSV', 'העלאת CSV')}
      </Button>
    </div>
  );
}

function UploadCenter({ copy, locale, notify }) {
  const [status, setStatus] = useState({ inputs: [] });
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState(true);

  const loadStatus = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/uploads/status`);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      setStatus(await response.json());
      setOnline(true);
    } catch {
      setStatus({ inputs: [] });
      setOnline(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  const inputs = normalizeRows(status.inputs);
  const groups = cadenceOrder
    .map((cadence) => ({
      cadence,
      inputs: inputs.filter((input) => input.cadence === cadence),
    }))
    .filter((group) => group.inputs.length > 0);

  return (
    <section className="page-workspace">
      <div className="page-header">
        <div>
          <h1>{pageText(locale, 'Upload', 'העלאה')}</h1>
          <p>
            {pageText(
              locale,
              'Three source data files come from the channel (programme lineup, historical spots, dayparts) plus the daily Wally ad file. Advertiser terms and the rate card are configuration. Extra columns in any file are kept, never discarded.',
              'שלושה קבצי נתונים מגיעים מהערוץ (לוח תוכניות, תשדירים היסטוריים, חלקי יום) ובנוסף קובץ הפרסומות היומי (Wally). תנאי המפרסמים וכרטיס התעריפים הם תצורה. עמודות נוספות בכל קובץ נשמרות, לא נמחקות.',
            )}
          </p>
        </div>
        <Button className="secondary-button compact" type="button" variant="outlined" onClick={loadStatus}>
          <RefreshCcw size={14} />
          {copy?.refresh || pageText(locale, 'Refresh', 'רענון')}
        </Button>
      </div>

      {loading && (
        <div className="upload-state">{pageText(locale, 'Loading data inputs...', 'טוען קלטי נתונים...')}</div>
      )}

      {!loading && !online && (
        <div className="upload-state error">
          <Database size={16} />
          {pageText(locale, 'The Kairos API is unavailable. Inputs cannot be shown.', 'ה־API של Kairos לא זמין. לא ניתן להציג קלטים.')}
        </div>
      )}

      {!loading && online && groups.length === 0 && (
        <div className="upload-state">{pageText(locale, 'No data inputs were reported by the API.', 'ה־API לא דיווח על קלטי נתונים.')}</div>
      )}

      {!loading && online && groups.map((group) => (
        <section className="page-panel" key={group.cadence}>
          <div className="panel-head">
            <div>
              <h2>{pageText(locale, cadenceMeta[group.cadence].en, cadenceMeta[group.cadence].he)}</h2>
              <small className="upload-group-note">
                {pageText(locale, cadenceMeta[group.cadence].noteEn, cadenceMeta[group.cadence].noteHe)}
              </small>
            </div>
            <span>{group.inputs.length} {pageText(locale, 'inputs', 'קלטים')}</span>
          </div>
          <div className="upload-grid">
            {group.inputs.map((input) => (
              <InputCard
                key={input.kind}
                input={input}
                locale={locale}
                onUploaded={loadStatus}
                notify={notify}
              />
            ))}
          </div>
        </section>
      ))}
    </section>
  );
}

export default UploadCenter;
