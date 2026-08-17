import React, { useEffect, useMemo, useState } from 'react';
import { Radar } from 'lucide-react';
import {
  Card, CardBody, EmptyState, ErrorState, InputControl, LoadingState, Status,
} from '../../studio';
import { Figure, Name, Prose } from '../../shell/bidi';
import { formatDay } from '../../shell/dates';
import { formatNumber, pageText } from '../../shell/surface-helpers';
import './forecast-stage.css';

// The rating forecast as a first-class stage: every programme of the day, the
// expected rating with its honest range, the drivers behind the number, and
// the historical mean beside it, because that mean is what this product
// priced on before the model existed, and the backtest's verdict on which of
// the two is more accurate is printed at the top rather than implied.

const BASE = '/api/forecast';

async function readJson(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.reason_he || body.detail || `${response.status}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

function loadSchedule(date) {
  return fetch(`${BASE}/schedule?date=${encodeURIComponent(date)}`).then(readJson);
}

function loadAccuracy() {
  return fetch(`${BASE}/accuracy`).then(readJson);
}

function tvr(value, locale) {
  if (value === null || value === undefined) return '-';
  return formatNumber(Math.round(Number(value) * 100) / 100, locale);
}

function AccuracyStrip({ accuracy, locale }) {
  // 403 means the measurement rides the model console's company wall, a
  // stated boundary, not an error.
  if (accuracy && accuracy.walled) {
    return (
      <p className="fcs-verdict fcs-verdict-quiet">
        {pageText(
          locale,
          'The accuracy measurement is company-only; the forecast itself is not.',
          'מדידת הדיוק נגישה לחשבונות חברה בלבד; התחזית עצמה פתוחה.',
        )}
      </p>
    );
  }
  if (!accuracy || !accuracy.available || !accuracy.overall) return null;
  const overall = accuracy.overall;
  const verdict = accuracy.verdict || {};
  return (
    <Card as="section" className="fcs-accuracy" aria-label={pageText(locale, 'Measured accuracy', 'דיוק מדוד')}>
      <CardBody>
        <div className="fcs-accuracy-figures">
          <div>
            <span className="fcs-cell-label">{pageText(locale, 'Log-space RMSE, model vs pre-model mean', 'RMSE במרחב הלוג, מודל מול הממוצע שקדם לו')}</span>
            <Figure>{`${overall.log_rmse} · ${overall.historical_log_rmse}`}</Figure>
          </div>
          <div>
            <span className="fcs-cell-label">{pageText(locale, 'MAE in rating points, model vs pre-model mean', 'MAE בנקודות רייטינג, מודל מול הממוצע שקדם לו')}</span>
            <Figure>{`${overall.mae} · ${overall.historical_mae}`}</Figure>
          </div>
          <div>
            <span className="fcs-cell-label">{pageText(locale, 'Band coverage at the 0.8 level', 'כיסוי הרצועה ברמת 0.8')}</span>
            <Figure>{String(overall.interval_coverage)}</Figure>
          </div>
          <div>
            <span className="fcs-cell-label">{pageText(locale, 'Observations scored', 'תצפיות שנמדדו')}</span>
            <Figure>{formatNumber(overall.interval_n, locale)}</Figure>
          </div>
        </div>
        {verdict.headline_he || verdict.headline_en ? (
          <Prose className="fcs-verdict">
            {pageText(locale, verdict.headline_en || '', verdict.headline_he || verdict.headline_en || '')}
          </Prose>
        ) : null}
      </CardBody>
    </Card>
  );
}

function DriverRows({ drivers, locale }) {
  if (!drivers || drivers.length === 0) {
    return (
      <p className="fcs-quiet">
        {pageText(locale, 'No driver decomposition is available for this row.', 'אין פירוק גורמים זמין לשורה הזאת.')}
      </p>
    );
  }
  return (
    <ul className="fcs-drivers">
      {drivers.map((driver) => (
        <li key={driver.key}>
          <Name>{pageText(locale, driver.label_en || driver.key, driver.label_he || driver.label_en || driver.key)}</Name>
          <Figure>
            {driver.kind === 'base'
              ? tvr(driver.value_tvr, locale)
              : `×${formatNumber(driver.multiplier, locale)}`}
          </Figure>
        </li>
      ))}
    </ul>
  );
}

function ProgrammeDetail({ row, locale }) {
  const interval = row.interval || {};
  const history = row.history || {};
  return (
    <Card as="aside" className="fcs-detail" aria-label={pageText(locale, 'Forecast detail', 'פירוט התחזית')}>
      <CardBody>
        <h4><Name>{row.title}</Name></h4>
        <dl className="fcs-detail-figures">
          <dt>{pageText(locale, 'Expected rating', 'רייטינג צפוי')}</dt>
          <dd><Figure>{tvr(row.expected_tvr, locale)}</Figure></dd>
          <dt>{pageText(locale, `Range at the ${interval.level || 0.8} level`, `טווח ברמת ${interval.level || 0.8}`)}</dt>
          <dd>
            {interval.available
              ? <Figure>{`${tvr(interval.low, locale)}–${tvr(interval.high, locale)}`}</Figure>
              : <span className="fcs-quiet">{pageText(locale, 'No honest band', 'אין רצועה כנה')}</span>}
          </dd>
          <dt>{history.label_he || pageText(locale, 'Historical mean', 'ממוצע היסטורי')}</dt>
          <dd><Figure>{tvr(history.historical_tvr, locale)}</Figure></dd>
        </dl>
        <h5>{pageText(locale, 'The drivers behind the number', 'הגורמים מאחורי המספר')}</h5>
        <DriverRows drivers={row.drivers} locale={locale} />
        {row.not_applied && row.not_applied.length > 0 ? (
          <p className="fcs-quiet">
            {pageText(locale, 'Held out of this forecast: ', 'לא הופעלו בתחזית הזאת: ')}
            {row.not_applied.map((item) => item.label_he || item.label_en || item.key || String(item)).join(', ')}
          </p>
        ) : null}
        {interval.method_he ? <Prose className="fcs-method">{interval.method_he}</Prose> : null}
      </CardBody>
    </Card>
  );
}

export default function ForecastStageWorkspace({ locale = 'he', refreshKey = 0 }) {
  const [date, setDate] = useState('2024-11-01');
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState('');
  const [accuracy, setAccuracy] = useState(null);
  const [openIndex, setOpenIndex] = useState(0);

  useEffect(() => {
    let alive = true;
    setPayload(null);
    setError('');
    loadSchedule(date).then(
      (body) => { if (alive) { setPayload(body); setOpenIndex(0); } },
      (failure) => { if (alive) setError(failure.message); },
    );
    return () => { alive = false; };
  }, [date, refreshKey]);

  useEffect(() => {
    let alive = true;
    loadAccuracy().then(
      (body) => { if (alive) setAccuracy(body); },
      (failure) => { if (alive) setAccuracy({ walled: failure.status === 403 }); },
    );
    return () => { alive = false; };
  }, [refreshKey]);

  const day = payload && payload.available !== false && payload.days && payload.days[0];
  const programmes = useMemo(() => (day ? day.programmes || [] : []), [day]);
  const summary = day && day.summary;
  const openRow = programmes[openIndex] || null;

  return (
    <section className="page-workspace fcs-workspace" aria-label={pageText(locale, 'Rating forecast', 'תחזית רייטינג')}>
      <header className="fcs-head">
        <div>
          <h2>{pageText(locale, 'Rating forecast', 'תחזית רייטינג')}</h2>
          <p>
            {pageText(
              locale,
              'Every programme of the day, the expected rating with its honest range, the drivers behind the number, and the pre-model historical mean beside it. The measured verdict on which is more accurate is printed above the table, not implied.',
              'כל תוכנית ביום, הרייטינג הצפוי עם הטווח הכן שלו, הגורמים מאחורי המספר, והממוצע ההיסטורי שקדם למודל לצידו. פסק הדין המדוד על מי מדויק יותר מודפס מעל הטבלה, לא נרמז.',
            )}
          </p>
        </div>
        <div className="fcs-controls">
          <InputControl
            type="date"
            value={date}
            onChange={(event) => setDate(event.target.value)}
            aria-label={pageText(locale, 'Broadcast day', 'יום שידור')}
          />
        </div>
      </header>

      <AccuracyStrip accuracy={accuracy} locale={locale} />

      {error ? (
        <ErrorState
          title={pageText(locale, 'The forecast could not be read', 'לא ניתן היה לקרוא את התחזית')}
          description={error}
        />
      ) : payload === null ? (
        <LoadingState title={pageText(locale, 'Reading the forecast', 'קורא את התחזית')} />
      ) : !day ? (
        <EmptyState
          title={pageText(locale, 'No schedule covers this day', 'אין לוח תוכניות שמכסה את היום הזה')}
          description={(payload && payload.reason_he) || ''}
        />
      ) : (
        <>
          <p className="fcs-context">
            <Figure>{formatDay(day.date)}</Figure>
            {' · '}
            <Name>{payload.channel}</Name>
            {summary ? (
              <>
                {' · '}
                {pageText(
                  locale,
                  `${summary.n_forecast} of ${summary.n} programmes forecast · mean expected ${tvr(summary.mean_expected_tvr, locale)} vs historical ${tvr(summary.mean_historical_tvr, locale)}`,
                  `${summary.n_forecast} מתוך ${summary.n} תוכניות נחזו · צפוי ממוצע ${tvr(summary.mean_expected_tvr, locale)} מול היסטורי ${tvr(summary.mean_historical_tvr, locale)}`,
                )}
              </>
            ) : null}
            {payload.audience_basis ? (
              <>
                {' · '}
                <Name>{payload.audience_basis.audience}</Name>
              </>
            ) : null}
          </p>

          <div className="fcs-body">
            <Card as="section" className="fcs-table-card" aria-label={pageText(locale, 'The programmes', 'התוכניות')}>
              <CardBody>
                <div className="fcs-table-scroll">
                  <table className="fcs-table">
                    <thead>
                      <tr>
                        <th>{pageText(locale, 'Clock', 'שעה')}</th>
                        <th>{pageText(locale, 'Programme', 'תוכנית')}</th>
                        <th>{pageText(locale, 'Expected', 'צפוי')}</th>
                        <th>{pageText(locale, 'Range', 'טווח')}</th>
                        <th>{pageText(locale, 'Historical', 'היסטורי')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {programmes.map((row, index) => (
                        <tr
                          key={`${row.start_seconds}-${index}`}
                          data-open={index === openIndex ? 'true' : 'false'}
                          onClick={() => setOpenIndex(index)}
                        >
                          <td><Figure>{row.start_clock}</Figure></td>
                          <td className="fcs-title"><Name>{row.title}</Name></td>
                          <td><Figure>{tvr(row.expected_tvr, locale)}</Figure></td>
                          <td>
                            {row.interval && row.interval.available
                              ? <Figure>{`${tvr(row.interval.low, locale)}–${tvr(row.interval.high, locale)}`}</Figure>
                              : <span className="fcs-quiet">-</span>}
                          </td>
                          <td><Figure>{tvr(row.history && row.history.historical_tvr, locale)}</Figure></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardBody>
            </Card>
            {openRow ? <ProgrammeDetail row={openRow} locale={locale} /> : null}
          </div>

          {payload.audience_basis && payload.audience_basis.measure_he ? (
            <p className="fcs-footnote">
              <Radar size={14} aria-hidden="true" />
              {payload.audience_basis.measure_he}
            </p>
          ) : null}
        </>
      )}
    </section>
  );
}
