import React from 'react';
import { Button, Tooltip } from '@mui/material';
import { Download } from 'lucide-react';
import { Numeric, formatNumber, pageText } from '../shell/format';
import { normalizeRows } from '../shell/plan-model';
import {
  downloadComplianceReport,
  downloadDataQualityReport,
  downloadRevenueReport,
  downloadScheduleCsv,
  downloadSpotsLedgerCsv,
} from '../shell/downloads';
import { DataTable, PageHeader, StatusBadge } from '../shell/primitives';

export function ReportsPage({ reports, files, overview, copy, locale, notify }) {
  const reportRows = normalizeRows(reports.reports);
  const fileRows = normalizeRows(files.files);

  // The daily spot ledger card carries the same meta whichever id the API
  // publishes it under; unused aliases never render.
  const spotsLedgerMeta = {
    titleEn: 'Daily spot ledger', titleHe: 'יומן ספוטים יומי',
    ownerEn: 'Revenue', ownerHe: 'הכנסות',
    descEn: 'Every priced spot with its premium and revenue, plus every dropped spot with its reason.',
    descHe: 'כל ספוט מתומחר עם הפרמיה וההכנסה שלו, וכל ספוט שנשמט עם הסיבה לכך.',
    download: () => downloadSpotsLedgerCsv(locale, notify),
  };

  // The API sends English-only titles/owners with stable report ids. Each id maps
  // to its localized title, owner, one-line description, and the downloader that
  // fetches its REAL content as a CSV. Any new id the API adds falls back to the
  // raw payload text with no download.
  const REPORT_META = {
    'weekly-plan': {
      titleEn: 'Weekly traffic plan', titleHe: 'תוכנית טראפיק שבועית',
      ownerEn: 'Traffic', ownerHe: 'טראפיק',
      descEn: 'Every scheduled break for the week, per channel and day.',
      descHe: 'כל הברייקים המשובצים לשבוע, לכל ערוץ ולכל יום.',
      download: () => downloadScheduleCsv(locale, notify, overview?.schedule_freshness),
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
      descHe: 'ההכנסה, השימור ומספר הברייקים החזויים לכל יום בשבוע.',
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
    <section className="page-workspace">
      <PageHeader
        locale={locale}
        titleEn="Reports"
        titleHe="דוחות"
        bodyEn="Download the weekly plan, compliance, revenue and source-audit reports for sales, operations and legal review. Each report is a live CSV built from the current data."
        bodyHe="הורדת דוחות התוכנית השבועית, התאימות, ההכנסה ובקרת המקורות עבור מכירות, תפעול וייעוץ משפטי. כל דוח הוא קובץ CSV חי שנבנה מהנתונים הנוכחיים."
        action={
          <Button className="secondary-button" type="button" variant="outlined" onClick={downloadAll}>
            <Download size={14} />
            {pageText(locale, 'Download all', 'הורדת הכל')}
          </Button>
        }
      />
      <div className="report-grid">
        {reportRows.map((report) => {
          const info = meta(report);
          const downloadable = isDownloadable(report);
          return (
            <article className="report-card" key={report.id}>
              <div className="report-card-head">
                <strong>{info ? pageText(locale, info.titleEn, info.titleHe) : report.title}</strong>
                <span className="report-card-owner">{info ? pageText(locale, info.ownerEn, info.ownerHe) : report.owner}</span>
              </div>
              {info ? <p className="report-card-desc">{pageText(locale, info.descEn, info.descHe)}</p> : null}
              <div className="report-card-meta">
                <StatusBadge status={report.status} locale={locale} />
                <small>{formatNumber(report.rows, locale)} {pageText(locale, 'rows', 'שורות')}</small>
              </div>
              <Tooltip title={downloadable ? '' : pageText(locale, 'This report has no data to download yet.', 'לדוח זה אין עדיין נתונים להורדה.')} arrow placement="bottom">
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
          <h2>{pageText(locale, 'Source package', 'חבילת מקורות')}</h2>
          <span><Numeric>{fileRows.filter((file) => file.exists).length} / {fileRows.length}</Numeric></span>
        </div>
        <DataTable
          locale={locale}
          emptyLabel={pageText(locale, 'No report source files were found.', 'לא נמצאו קבצי מקור לדוחות.')}
          rows={fileRows}
          columns={[
            { key: 'path', label: pageText(locale, 'File', 'קובץ') },
            { key: 'exists', label: pageText(locale, 'State', 'מצב'), status: true, minWidth: 104, flex: 0.45, render: (row) => <StatusBadge status={row.exists ? 'ready' : 'error'} locale={locale} mode="cell" /> },
            { key: 'size', label: pageText(locale, 'Size', 'גודל'), render: (row) => `${formatNumber(Number(row.size || 0) / 1024, locale)} KB` },
            { key: 'modified', label: pageText(locale, 'Modified', 'עודכן'), render: (row) => (row.modified ? new Date(row.modified).toLocaleString(locale === 'he' ? 'he-IL' : 'en-US') : '-') },
          ]}
        />
      </section>
    </section>
  );
}

export default ReportsPage;
