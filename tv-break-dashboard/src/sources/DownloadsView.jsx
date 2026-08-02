import React, { useState } from 'react';
import { Button, Tooltip } from '@mui/material';
import { Download, Rows3 } from 'lucide-react';
import { Numeric, formatNumber, pageText } from '../shell/format';
import { normalizeRows } from '../shell/plan-model';
import {
  downloadComplianceReport,
  downloadDataQualityReport,
  downloadRevenueReport,
  downloadScheduleCsv,
  downloadSpotsLedgerCsv,
} from '../shell/downloads';
import { StatusBadge } from '../shell/primitives';
import { serverText, text } from './sources-copy';
import ReportRowsDrawer from './ReportRowsDrawer';

// The declared basis, printed on the report and not in a tooltip: the period
// the rows cover, the scope they are summed over, the file they are built
// from, and when that source last changed. Stripe attaches four facts to every
// report for exactly this reason, and a fact that cannot be computed is left
// out rather than filled in.
function factValue(fact, locale) {
  const value = locale === 'he' ? fact.value_he : fact.value_en;
  // A moment is sent as an ISO instant so it is unambiguous on the wire. It is
  // read by a person, so it is printed the way this locale writes a date.
  if (fact.code !== 'updated') return value;
  const when = new Date(value);
  return Number.isNaN(when.getTime()) ? value : when.toLocaleString(locale === 'he' ? 'he-IL' : 'en-US');
}

function Basis({ basis, locale }) {
  const facts = Array.isArray(basis) ? basis : [];
  if (facts.length === 0) return null;
  return (
    <dl className="report-basis">
      {facts.map((fact) => (
        <div key={fact.code}>
          <dt>{locale === 'he' ? fact.label_he : fact.label_en}</dt>
          <dd dir="auto">{factValue(fact, locale)}</dd>
        </div>
      ))}
    </dl>
  );
}

// The five reports, each built from the current data at the moment it is
// downloaded. The row count beside a card is the exact number of rows that
// file will carry, and it opens those rows, so the claim can be checked
// instead of believed.
export function DownloadsView({ reports, files, overview, locale, notify }) {
  const [rowsFor, setRowsFor] = useState(null);
  const reportRows = normalizeRows(reports && reports.reports);
  const fileRows = normalizeRows(files && files.files);

  // The daily spot ledger card carries the same meta whichever id the API
  // publishes it under; unused aliases never render.
  const spotsLedgerMeta = {
    titleEn: 'Daily spot ledger', titleHe: 'יומן ספוטים יומי',
    ownerEn: 'Revenue', ownerHe: 'הכנסות',
    descEn: 'Every priced spot with its premium and revenue, plus every dropped spot with its reason.',
    descHe: 'כל ספוט מתומחר עם הפרמיה וההכנסה שלו, וכל ספוט שנשמט עם הסיבה לכך.',
    download: () => downloadSpotsLedgerCsv(locale, notify),
  };

  // The API sends English-only titles and owners with stable report ids. Each id
  // maps to its localized title, owner, one-line description, and the downloader
  // that fetches its REAL content as a CSV. Any new id the API adds falls back to
  // the raw payload text with no download.
  const REPORT_META = {
    'weekly-plan': {
      titleEn: 'Weekly traffic plan', titleHe: 'תוכנית טראפיק שבועית',
      ownerEn: 'Traffic', ownerHe: 'טראפיק',
      descEn: 'Every scheduled break for the week, per channel and day.',
      descHe: 'כל הברייקים המשובצים לשבוע, לכל ערוץ ולכל יום.',
      download: () => downloadScheduleCsv(locale, notify, overview && overview.schedule_freshness),
    },
    compliance: {
      titleEn: 'Compliance and guardrails', titleHe: 'תאימות ובקרות',
      ownerEn: 'Legal / Ops', ownerHe: 'משפטי / תפעול',
      descEn: 'Every regulatory check against its limit, with the observed value and status.',
      descHe: 'כל בדיקת רגולציה מול המגבלה שלה, עם הערך שנמדד והסטטוס.',
      download: () => downloadComplianceReport(locale, notify),
    },
    revenue: {
      titleEn: 'Revenue forecast', titleHe: 'תחזית הכנסה',
      ownerEn: 'Revenue', ownerHe: 'הכנסות',
      descEn: 'Saved-plan revenue, retention and break count for each day on the plan.',
      descHe: 'ההכנסה, השימור ומספר הברייקים החזויים לכל יום בתוכנית.',
      download: () => downloadRevenueReport(locale, notify),
    },
    'data-quality': {
      titleEn: 'Source file audit', titleHe: 'בקרת קבצי מקור',
      ownerEn: 'Data', ownerHe: 'נתונים',
      descEn: 'The presence, size and last-modified time of every source file.',
      descHe: 'הקיום, הגודל ומועד העדכון האחרון של כל קובץ מקור.',
      download: () => downloadDataQualityReport(fileRows, locale, notify),
    },
    'daily-spots': spotsLedgerMeta,
    'spots-ledger': spotsLedgerMeta,
  };

  const meta = (report) => REPORT_META[report.id] || null;
  const isDownloadable = (report) => Boolean(meta(report)) && report.status !== 'empty' && Number(report.rows) > 0;

  async function downloadAll() {
    const ready = reportRows.filter(isDownloadable);
    if (ready.length === 0) {
      if (notify) notify('No reports are ready to download yet.', 'אין דוחות מוכנים להורדה עדיין.');
      return;
    }
    // Sequential so each file save settles before the next begins.
    for (const report of ready) {
      // eslint-disable-next-line no-await-in-loop
      await meta(report).download();
    }
  }

  return (
    <div className="sources-view">
      <div className="downloads-head">
        <p>{text('downloadsBody', locale)}</p>
        <Button className="secondary-button" type="button" variant="outlined" onClick={downloadAll}>
          <Download size={14} />
          {text('downloadAll', locale)}
        </Button>
      </div>
      <div className="report-grid">
        {reportRows.map((report) => {
          const info = meta(report);
          const downloadable = isDownloadable(report);
          const title = info ? pageText(locale, info.titleEn, info.titleHe) : report.title;
          return (
            <article className="report-card" key={report.id} data-report={report.id}>
              <div className="report-card-head">
                <strong>{title}</strong>
                <span className="report-card-owner">{info ? pageText(locale, info.ownerEn, info.ownerHe) : report.owner}</span>
              </div>
              {info ? <p className="report-card-desc">{pageText(locale, info.descEn, info.descHe)}</p> : null}
              <div className="report-card-meta">
                <StatusBadge status={report.status} locale={locale} />
                <Tooltip title={text('openReportRows', locale)} arrow placement="top">
                  <span>
                    <Button
                      className="link-figure"
                      type="button"
                      disabled={!Number(report.rows)}
                      onClick={() => setRowsFor({ ...report, title })}
                    >
                      <Rows3 size={12} />
                      <Numeric>{formatNumber(report.rows, locale)}</Numeric>
                      {text('reportRows', locale)}
                    </Button>
                  </span>
                </Tooltip>
              </div>
              {report.unit ? <p className="report-card-unit">{serverText(report.unit, locale)}</p> : null}
              <Basis basis={report.basis} locale={locale} />
              <Tooltip title={downloadable ? '' : text('reportEmpty', locale)} arrow placement="bottom">
                <span className="report-card-download-wrap">
                  <Button
                    className="report-card-download"
                    type="button"
                    variant="outlined"
                    size="small"
                    disabled={!downloadable}
                    onClick={() => info && info.download()}
                  >
                    <Download size={13} />
                    {pageText(locale, 'Download CSV', 'הורדת CSV')}
                  </Button>
                </span>
              </Tooltip>
            </article>
          );
        })}
      </div>
      <section className="page-panel">
        <div className="panel-head">
          <h2>{text('sourcePackage', locale)}</h2>
          <span>
            <Numeric>{formatNumber(fileRows.filter((file) => file.in_use).length, locale)}</Numeric> {text('readOfPresent', locale)} <Numeric>{formatNumber(fileRows.filter((file) => file.exists).length, locale)}</Numeric> {text('present', locale)}
          </span>
        </div>
        <ul className="downloads-file-list">
          {fileRows.map((file) => (
            <li key={file.path}>
              <span className="source-file-path" dir="ltr">{file.path}</span>
              <span className={file.in_use ? 'source-state ok' : 'source-state warn'}>
                {file.in_use ? text('fileYes', locale) : text('fileNo', locale)}
              </span>
            </li>
          ))}
        </ul>
      </section>

      {rowsFor ? (
        <ReportRowsDrawer report={rowsFor} files={files} locale={locale} onClose={() => setRowsFor(null)} />
      ) : null}
    </div>
  );
}

export default DownloadsView;
