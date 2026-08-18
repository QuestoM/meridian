import React, { useMemo } from 'react';
import { Button } from '../studio/actions';
import { Figure } from '../shell/bidi';
import { ArrowLeft, Coins, Layers, Receipt } from 'lucide-react';
import { pageText } from '../shell/format';
import SourceFileLink from './SourceFileLink';
import { NO_DRILL, basisPrefix, exactMoney, goToView, localized, periodNote, scopeNote, widerPeriod } from './clients-money-helpers';
import MoneyDetail from './MoneyDetail';
import { isolate } from '../shell/bidi';
import { droppingRuleLine } from './clients-rule-helpers';

// The analyst's surface: what each client delivered, gross and net of the
// agency rebate, with every figure opening the rows behind it.
//
// Three rules hold it honest and each is visible on screen rather than in a
// comment. The scope is printed with the totals, never in a tooltip. The period
// is stated as what it is, one broadcast day, with the path to widen it, because
// "last month" is not a period this data has. And the same rows are re-grouped
// four ways rather than re-summed, so a client total and an agency total are the
// same shekels counted along a different edge.
//
// Which row is open is a prop, not this component's own state. A client record
// opens its own money from outside this board, and a board that kept the key to
// itself could only be told to appear, never told which row to show.

// Each grouping asks its own question, because the answer under the headline is
// whatever the active grouping ranks. One hard-coded question over four rankings
// put an agency, a campaign and a break under the words "which client delivered
// the most", which is a false sentence attached to a true number on the surface
// the money question exists to answer.
const GROUPS = [
  {
    key: 'advertisers',
    field: 'advertiser',
    en: 'By client',
    he: 'לפי לקוח',
    columnEn: 'Client',
    columnHe: 'לקוח',
    questionEn: 'Which client delivered the most, gross and net of agency rebates',
    questionHe: 'איזה לקוח סיפק הכי הרבה, ברוטו ונטו אחרי רבייט הסוכנות',
  },
  {
    key: 'agencies',
    field: 'agency',
    en: 'By agency',
    he: 'לפי סוכנות',
    columnEn: 'Agency',
    columnHe: 'סוכנות',
    questionEn: 'Which agency delivered the most, gross and net of agency rebates',
    questionHe: 'איזו סוכנות סיפקה הכי הרבה, ברוטו ונטו אחרי רבייט הסוכנות',
  },
  {
    key: 'campaigns',
    field: 'campaign',
    en: 'By campaign',
    he: 'לפי קמפיין',
    columnEn: 'Campaign',
    columnHe: 'קמפיין',
    questionEn: 'Which campaign delivered the most, gross and net of agency rebates',
    questionHe: 'איזה קמפיין סיפק הכי הרבה, ברוטו ונטו אחרי רבייט הסוכנות',
  },
  {
    key: 'breaks',
    field: 'break_id',
    en: 'By break',
    he: 'לפי ברייק',
    columnEn: 'Break',
    columnHe: 'ברייק',
    questionEn: 'Which break delivered the most, gross and net of agency rebates',
    questionHe: 'איזה ברייק סיפק הכי הרבה, ברוטו ונטו אחרי רבייט הסוכנות',
  },
];

// Where the open row sits in the set it was opened from, so a reader can walk
// the whole ranking from inside one record instead of going back to find it.
function positionOf(rows, field, key) {
  const index = rows.findIndex((row) => String(row[field]) === key);
  return index < 0 ? null : { position: index + 1, total: rows.length };
}

function stepKey(rows, field, key, delta) {
  const index = rows.findIndex((row) => String(row[field]) === key);
  if (index < 0 || !rows.length) {
    return key;
  }
  return String(rows[(index + delta + rows.length) % rows.length][field]);
}

function Tile({ label, value, sub, icon: Icon, tone }) {
  return (
    <div className={`card card-dense card-body clients-tile ${tone || ''}`}>
      <span className="clients-tile-icon"><Icon size={16} strokeWidth={1.8} /></span>
      <span className="clients-tile-copy">
        <span className="clients-tile-label">{label}</span>
        <strong className="numeric"><Figure>{value}</Figure></strong>
        {sub ? <small>{sub}</small> : null}
      </span>
    </div>
  );
}

// The records the head of a drill can open, supplied by the workspace because
// the two indexes it needs are built from the client tree and the campaign
// board, which are that surface's reads and not this one's. An absent opener is
// an honest state and not a fault: the head stays a label wherever the object
// behind it cannot be reached.
// The dropped-spot line under the net tile. It names the rule, because a count
// beside the word "rule" tells a reader that money is missing and nothing about
// how to get it back — and the rule was never further away than one drawer, one
// campaign and one spot, which is exactly what made the omission easy to keep.
function dropSub(totals, locale) {
  const count = totals.dropped_by_frequency;
  const which = droppingRuleLine(totals.dropped_rules, locale);
  if (!which) {
    // No rule could be named — an unreadable rule file, say. The count is still
    // true and still worth stating; what is not known is not invented.
    return pageText(
      locale,
      `${count} spots dropped by a rule`,
      `${isolate(count)} תשדירים הוסרו על ידי כלל`,
    );
  }
  return pageText(locale, `${count} spots dropped. ${which}`, `${isolate(count)} תשדירים הוסרו. ${which}`);
}


export default function MoneyBoard({
  money,
  locale,
  drill = NO_DRILL,
  onDrill = () => {},
  onOpenClient,
  openers = {},
}) {
  const group = drill.group || NO_DRILL.group;
  const openKey = drill.key || '';
  const he = locale === 'he';

  const definition = GROUPS.find((entry) => entry.key === group) || GROUPS[0];
  const rows = useMemo(() => (money && money[group]) || [], [money, group]);
  const ranked = useMemo(
    () => [...rows].sort((left, right) => right.gross - left.gross),
    [rows],
  );
  const leader = ranked[0];
  const open = openKey ? rows.find((row) => String(row[definition.field]) === openKey) : null;

  if (!money) {
    return <div className="clients-loading">{pageText(locale, 'Loading the ledger', 'טוען את הספר')}</div>;
  }

  if (!money.available) {
    return (
      <section className="clients-empty">
        <Coins size={22} aria-hidden="true" />
        <strong>{pageText(locale, 'No priced day to read', 'אין יום מתומחר לקריאה')}</strong>
        <p>{localized(money, 'reason', locale)}</p>
        <p className="clients-empty-path">
          {pageText(
            locale,
            'A daily spot file prices a day and fills this board with the real ledger.',
            'קובץ תשדירים יומי מתמחר יום וממלא את הלוח הזה מהספר האמיתי.',
          )}
        </p>
        <Button type="button" className="clients-primary" onClick={() => goToView('Data')}>
          {pageText(locale, 'Go to Data and upload one', 'עברו למסך הנתונים והעלו קובץ')}
        </Button>
      </section>
    );
  }

  const totals = money.totals;
  const basis = money.basis;

  return (
    <section className="clients-money">
      <div className="card card-dense card-body clients-answer">
        <p className="clients-answer-question">
          {pageText(locale, definition.questionEn, definition.questionHe)}
        </p>
        {leader ? (
          <Button
            type="button"
            className="clients-answer-line"
            onClick={() => onDrill({ group, key: String(leader[definition.field]) })}
          >
            <strong>{leader[definition.field] || pageText(locale, 'Unnamed', 'ללא שם')}</strong>
            <Figure className="numeric">{exactMoney(leader.gross, locale)}</Figure>
            <small>{pageText(locale, 'gross', 'ברוטו')}</small>
            <Figure className="numeric">{exactMoney(leader.net, locale)}</Figure>
            <small>{pageText(locale, 'net after rebates', 'נטו אחרי רבייט')}</small>
          </Button>
        ) : null}
        <p className="clients-basis">
          {basisPrefix(basis, locale)}
          <SourceFileLink name={basis && basis.file} locale={locale} />
        </p>
        <p className="clients-basis-note">{periodNote(basis, locale)}</p>
        <p className="clients-basis-note">{scopeNote(basis, locale)}</p>
        <p className="clients-basis-path">
          {widerPeriod(basis, locale)}
          <Button type="button" className="clients-inline-action" onClick={() => goToView('Data')}>
            {pageText(locale, 'Open Data', 'פתחו את מסך הנתונים')}
          </Button>
        </p>
      </div>

      <div className="clients-tiles">
        <Tile
          label={pageText(locale, 'Gross', 'ברוטו')}
          value={exactMoney(totals.gross, locale)}
          sub={pageText(locale, `${totals.spots} priced spots`, `${isolate(totals.spots)} תשדירים מתומחרים`)}
          icon={Coins}
        />
        <Tile
          label={pageText(locale, 'Agency rebates', 'רבייט סוכנויות')}
          value={exactMoney(totals.rebates, locale)}
          sub={pageText(locale, 'reporting only, nothing is invoiced', 'לדיווח בלבד, דבר אינו מחויב')}
          icon={Receipt}
        />
        <Tile
          label={pageText(locale, 'Net after rebates', 'נטו אחרי רבייט')}
          value={exactMoney(totals.net, locale)}
          sub={dropSub(totals, locale)}
          icon={Layers}
          tone="net"
        />
      </div>

      <div className="clients-group-tabs" role="tablist">
        {GROUPS.map((entry) => (
          <Button
            key={entry.key}
            type="button"
            role="tab"
            aria-selected={entry.key === group}
            className={entry.key === group ? 'active' : ''}
            onClick={() => onDrill({ group: entry.key, key: '' })}
          >
            {pageText(locale, entry.en, entry.he)}
          </Button>
        ))}
        <span className="clients-group-count">
          {pageText(locale, `${rows.length} rows, same ledger`, `${isolate(rows.length)} שורות, אותו ספר`)}
        </span>
      </div>

      {open ? (
        <div className="clients-drill">
          <Button type="button" className="clients-back" onClick={() => onDrill({ group, key: '' })}>
            <ArrowLeft size={14} aria-hidden="true" />
            {pageText(locale, 'All rows', 'כל השורות')}
          </Button>
          <MoneyDetail
            money={money}
            row={open}
            field={definition.field}
            locale={locale}
            position={positionOf(ranked, definition.field, openKey)}
            onStep={(delta) => onDrill({ group, key: stepKey(ranked, definition.field, openKey, delta) })}
            onOpenBreak={(breakId) => onDrill({ group: 'breaks', key: breakId })}
            onOpenCampaign={(name) => onDrill({ group: 'campaigns', key: name })}
            openers={{ ...openers, onOpenClient }}
          />
        </div>
      ) : null}

      {!open && openKey ? (
        <div className="clients-drill clients-drill-missing">
          <Button type="button" className="clients-back" onClick={() => onDrill({ group, key: '' })}>
            <ArrowLeft size={14} aria-hidden="true" />
            {pageText(locale, 'All rows', 'כל השורות')}
          </Button>
          <p className="clients-reason">
            {pageText(
              locale,
              `${openKey} has no row in the ledger this board is reading, so there is nothing to open behind it.`,
              `ל${isolate(openKey)} אין שורה בספר שהלוח הזה קורא, ולכן אין מה לפתוח מאחוריו.`,
            )}
          </p>
          <p className="clients-basis-note">
            {basisPrefix(basis, locale)}
            <SourceFileLink name={basis && basis.file} locale={locale} />
          </p>
        </div>
      ) : null}

      {!open && !openKey ? (
        <table className="clients-table">
          <thead>
            <tr>
              <th scope="col">{pageText(locale, 'Rank', 'דירוג')}</th>
              <th scope="col">{pageText(locale, definition.columnEn, definition.columnHe)}</th>
              <th scope="col" className="numeric-col">{pageText(locale, 'Gross', 'ברוטו')}</th>
              <th scope="col" className="numeric-col">{pageText(locale, 'Rebate', 'רבייט')}</th>
              <th scope="col" className="numeric-col">{pageText(locale, 'Net', 'נטו')}</th>
              <th scope="col" className="numeric-col">{pageText(locale, 'Share', 'נתח')}</th>
              <th scope="col" className="numeric-col">{pageText(locale, 'Spots', 'תשדירים')}</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((row) => (
              <tr
                key={String(row[definition.field])}
                className="clients-row-open"
                onClick={(event) => {
                  // The name stays the real control — it is what a keyboard and
                  // a screen reader reach, and the row is deliberately not a tab
                  // stop, because making every row one would double the length
                  // of the whole table for anyone travelling it by key. So the
                  // row only forwards clicks the name did not already take.
                  if (event.target.closest('button, a, input, [role=button]')) return;
                  // A click that ends a text selection is a selection, not a
                  // navigation. Reading a figure out of a table should not move
                  // the reader off it.
                  if (String(window.getSelection() || '')) return;
                  onDrill({ group, key: String(row[definition.field]) });
                }}
              >
                <td className="numeric"><Figure>{row.rank}</Figure></td>
                <td>
                  <Button type="button" className="clients-link" onClick={() => onDrill({ group, key: String(row[definition.field]) })}>
                    {String(row[definition.field]) || pageText(locale, 'Unnamed', 'ללא שם')}
                  </Button>
                </td>
                <td className="numeric"><Figure>{exactMoney(row.gross, locale)}</Figure></td>
                <td className="numeric"><Figure>{exactMoney(row.rebates, locale)}</Figure></td>
                <td className="numeric"><Figure>{exactMoney(row.net, locale)}</Figure></td>
                <td className="numeric"><Figure>{(row.share_of_gross * 100).toFixed(2)}%</Figure></td>
                <td className="numeric"><Figure>{row.spots}</Figure></td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </section>
  );
}
