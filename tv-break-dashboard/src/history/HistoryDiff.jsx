import React from 'react';
import { pageText } from '../shell/format';
import { Code, Name } from '../shell/bidi';
import { changeRows } from './history-fields';
import { FILE_LABELS, pair } from './history-labels';
import { rowIdentity } from './history-rows';

// What restoring a point would change, per file. This is the preview Figma's
// version history gives before a restore: the current value beside the value at
// that point, so the decision is made on the difference rather than on a date.
//
// The direction is fixed and it is the direction that matters: "current" is what
// the operator has now and "at this point" is what a restore would write. Both
// columns are read from the store; nothing here computes a value.
//
// A whole row that would be added or removed is not a field change, so it is
// read as an identity rather than as a table: what the row is, then the parts
// that tell it from its neighbours, both chosen by history-rows.js in the
// reader's own language. Printing the record instead is the defect this closed:
// measured on restore point e105c8d1da22, seven added restrictions each printed
// as {"constraint_id":"04e955157d4c","scope_t… at seventy-eight characters, so
// the only thing telling one from the next was a twelve-character record id and
// everything a person decides on was past the cut.
//
// A field whose value is itself a set of values is the same defect wearing a
// table: the rate card arrives as one nested object and printed as one cut
// record. history-fields.js reads a change into the rows a person decides on,
// so nothing on this surface is a dumped record any more.
//
// Every cell a store can write into is isolated rather than forced left to
// right, so a Hebrew channel name and a negative number are both read the way
// they were written.

function TriRow({ head, field, cur, ver }) {
  return (
    <div className={`hist-diff-row${head ? ' head' : ''}`}>
      <span className="hist-diff-field">{head ? field : <bdi>{field}</bdi>}</span>
      <span className="hist-diff-cur">{head ? cur : <bdi>{cur}</bdi>}</span>
      <span className="hist-diff-ver">{head ? ver : <bdi>{ver}</bdi>}</span>
    </div>
  );
}

// The rows one changed record reads as, in order, each already in words.
function FieldRows({ file, changed, locale }) {
  const rows = changed.flatMap((row) => changeRows(file, row, locale)
    .map((out) => ({ ...out, id: String((row && row.id) ?? '') })));
  return rows.map((row, index) => (
    <TriRow
      key={`${index}-${row.key}`}
      field={row.id ? `${row.id} / ${row.field}` : row.field}
      cur={row.cur}
      ver={row.ver}
    />
  ));
}

// One row, identified. dir is auto on everything a store can write Hebrew into,
// which is the title, the note and every scope value; only the record id and
// the dates and clocks are isolated left to right.
function RowChip({ file, item, locale, tone }) {
  const identity = rowIdentity(file, item, locale);
  return (
    <li className={`hist-diff-chip ${tone}`}>
      <span className="hist-diff-chip-name">
        {identity.label ? <span className="hist-diff-chip-key">{`${identity.label} `}</span> : null}
        <Name>{identity.title}</Name>
      </span>
      {identity.parts.length ? (
        <span className="hist-diff-chip-parts">
          {identity.parts.map((part) => (
            <span className="hist-diff-chip-part" key={part.key}>
              <span className="hist-diff-chip-key">{part.label}</span>
              {part.values.map((value, index) => (
                part.ltr ? <Code key={index}>{value}</Code> : <Name key={index}>{value}</Name>
              ))}
            </span>
          ))}
        </span>
      ) : null}
    </li>
  );
}

function ChipList({ title, items, file, locale, tone }) {
  const list = Array.isArray(items) ? items : [];
  if (!list.length) return null;
  return (
    <div className="hist-diff-sub">
      <span className="hist-diff-sub-h">{title}</span>
      <ul className="hist-diff-chips">
        {list.map((item, index) => <RowChip file={file} item={item} locale={locale} tone={tone} key={index} />)}
      </ul>
    </div>
  );
}

export function fileHasChanges(detail) {
  if (!detail || typeof detail !== 'object') return false;
  const changed = Array.isArray(detail.changed) ? detail.changed.length : 0;
  const added = Array.isArray(detail.added) ? detail.added.length : 0;
  const removed = Array.isArray(detail.removed) ? detail.removed.length : 0;
  return changed + added + removed > 0;
}

function FileDiff({ file, detail, locale }) {
  const headRow = (
    <TriRow
      head
      field={pageText(locale, 'Field', 'שדה')}
      cur={pageText(locale, 'Now', 'עכשיו')}
      ver={pageText(locale, 'At this point', 'בנקודה זו')}
    />
  );
  const changed = Array.isArray(detail.changed) ? detail.changed : [];
  let body;
  if (file === 'settings') {
    body = (
      <div className="hist-diff-grid">
        {headRow}
        <FieldRows file={file} changed={changed} locale={locale} />
      </div>
    );
  } else if (file === 'advertisers') {
    const groups = [];
    const index = new Map();
    changed.forEach((row) => {
      const name = String((row && row.advertiser) || '');
      if (!index.has(name)) {
        index.set(name, groups.length);
        groups.push({ name, rows: [] });
      }
      groups[index.get(name)].rows.push(row);
    });
    body = (
      <div className="hist-diff-store">
        <ChipList title={pageText(locale, 'Added', 'נוספו')} items={detail.added} file={file} locale={locale} tone="add" />
        <ChipList title={pageText(locale, 'Removed', 'הוסרו')} items={detail.removed} file={file} locale={locale} tone="remove" />
        {groups.map((group) => (
          <div className="hist-diff-adv" key={group.name || 'row'}>
            <span className="hist-diff-adv-h"><Name>{group.name}</Name></span>
            <div className="hist-diff-grid">
              {headRow}
              <FieldRows file={file} changed={group.rows} locale={locale} />
            </div>
          </div>
        ))}
      </div>
    );
  } else {
    body = (
      <div className="hist-diff-store">
        <ChipList title={pageText(locale, 'Added', 'נוספו')} items={detail.added} file={file} locale={locale} tone="add" />
        <ChipList title={pageText(locale, 'Removed', 'הוסרו')} items={detail.removed} file={file} locale={locale} tone="remove" />
        {changed.length ? (
          <div className="hist-diff-grid">
            {headRow}
            <FieldRows file={file} changed={changed} locale={locale} />
          </div>
        ) : null}
      </div>
    );
  }
  return (
    <section className="hist-diff-file">
      <h5 className="hist-diff-file-h">{pair(FILE_LABELS, file, locale) || file}</h5>
      {body}
    </section>
  );
}

export default function HistoryDiff({ diff, locale }) {
  const files = Object.keys(FILE_LABELS).filter((file) => diff && diff[file] && fileHasChanges(diff[file]));
  if (!files.length) {
    return <p className="hist-empty">{pageText(locale, 'Nothing would change. This point matches the current state.', 'שום דבר לא ישתנה. הנקודה הזו זהה למצב הנוכחי.')}</p>;
  }
  return <div className="hist-diff">{files.map((file) => <FileDiff key={file} file={file} detail={diff[file]} locale={locale} />)}</div>;
}
