import React, { useEffect, useState } from 'react';
import { Button, TextField } from '@mui/material';
import { ChevronDown, ChevronUp, History, Lock, RotateCcw } from 'lucide-react';
import { Numeric, formatCurrency, formatNumber, pageText } from '../../shell/format';
import { formatStamp } from '../../shell/dates';
import { Figure, Name } from '../../shell/bidi';
import { collapseWarning, diffReason, scopeLine } from './plan-week-model';

// Step four: freeze the plan under a name.
//
// The weekly plan was the one operational artifact nothing versioned, so the
// word publish appeared zero times in the backend and a run overwrote whatever
// the last run produced. A plan version is an internal freeze, per the owner
// ruling: a planner names it alone, it records who and when, it diffs against
// the one before it, and it restores byte for byte.
//
// Restore is destructive to the live plan, so it asks first and says exactly
// what it will do, and the plan it replaces is frozen automatically so the
// rollback is itself reversible.

function VersionRow({ version, locale, selected, canEdit, onSelect, onDiff, onRestore }) {
  const owned = version.summary?.owned || {};
  return (
    <div className={`plan-version-row${selected ? ' is-selected' : ''}`}>
      <button type="button" className="plan-version-main" onClick={() => onSelect(version.version_id)}>
        <Name className="plan-version-name">{version.name}</Name>
        <span className="plan-version-meta">
          <Numeric>{formatStamp(version.created_at) || version.created_at}</Numeric>
          {' · '}
          <Name>{version.actor}</Name>
        </span>
        <span className="plan-version-figures">
          <Numeric>{formatNumber(owned.breaks, locale)}</Numeric>
          <small>{pageText(locale, 'breaks', 'ברייקים')}</small>
          <Numeric>{formatCurrency(owned.revenue, locale)}</Numeric>
        </span>
      </button>
      <div className="plan-version-actions">
        <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => onDiff(version.version_id)}>
          <History size={13} />
          {pageText(locale, 'What changed', 'מה השתנה')}
        </Button>
        <Button
          className="secondary-button compact"
          type="button"
          variant="outlined"
          disabled={!canEdit}
          onClick={() => onRestore(version)}
        >
          <RotateCcw size={13} />
          {pageText(locale, 'Roll back to this', 'חזרה לגרסה הזאת')}
        </Button>
      </div>
    </div>
  );
}

export function PublishPanel({
  locale,
  words,
  versions,
  live,
  canEdit,
  canEditReason,
  name,
  note,
  publishState,
  publishError,
  selectedId,
  diff,
  onNameChange,
  onNoteChange,
  onPublish,
  onSelect,
  onDiff,
  onRestore,
}) {
  const alreadyFrozen = Boolean(live?.frozen_as);
  const collapse = collapseWarning(live);
  const [collapseConfirmed, setCollapseConfirmed] = useState(false);
  const frozenVersion = alreadyFrozen ? versions.find((item) => item.version_id === live.frozen_as) : null;
  const selectedIndex = selectedId ? versions.findIndex((item) => item.version_id === selectedId) : -1;
  // Walking the set selects and opens in one move, because a counter that only
  // moves a highlight would make the person click twice for every step.
  const walkTo = (index) => {
    const next = versions[index];
    if (!next) return;
    onSelect(next.version_id);
    onDiff(next.version_id);
  };
  const collapseKey = `${collapse.against_version_id || 'none'}:${collapse.current?.breaks ?? 'unknown'}:${collapse.current?.revenue ?? 'unknown'}`;
  useEffect(() => setCollapseConfirmed(false), [collapseKey]);

  return (
    <section className="plan-section" aria-labelledby="plan-publish-title">
      <div className="plan-section-head">
        <div>
          <h2 id="plan-publish-title">{pageText(locale, 'Freeze this plan', 'הקפאת התוכנית')}</h2>
          <p>
            {pageText(
              locale,
              'A frozen plan stops moving. It keeps its own copy of the week, who froze it and when, so everyone downstream can name the plan they are reading.',
              'תוכנית מוקפאת מפסיקה לזוז. היא שומרת עותק משלה של השבוע, מי הקפיא אותה ומתי, כדי שכל מי שקורא אותה בהמשך יוכל לנקוב בשם התוכנית שלפניו.',
            )}
          </p>
        </div>
      </div>

      {!canEdit && canEditReason && (
        <p className="plan-note plan-note-amber" role="status">{canEditReason}</p>
      )}

      {live?.exists === false ? (
        <p className="plan-note plan-note-amber" role="status">
          {pageText(
            locale,
            'There is no saved plan to freeze yet. Run the weekly plan first.',
            'אין עדיין תוכנית שמורה להקפאה. הריצו קודם את התוכנית השבועית.',
          )}
        </p>
      ) : (
        <>
          {collapse.collapsed && (
            <div className="plan-note plan-note-amber plan-collapse-warning" role="alert">
              <p>
                {collapse.previous
                  ? pageText(
                    locale,
                    `This plan has collapsed to ${formatNumber(collapse.current?.breaks, locale)} breaks and ${formatCurrency(collapse.current?.revenue, locale)} on your channel. The newest frozen plan had ${formatNumber(collapse.previous?.breaks, locale)} breaks and ${formatCurrency(collapse.previous?.revenue, locale)}.`,
                    `התוכנית הזאת קרסה ל־${formatNumber(collapse.current?.breaks, locale)} ברייקים ול־${formatCurrency(collapse.current?.revenue, locale)} בערוץ שלכם. בתוכנית המוקפאת החדשה ביותר היו ${formatNumber(collapse.previous?.breaks, locale)} ברייקים ו־${formatCurrency(collapse.previous?.revenue, locale)}.`,
                  )
                  : pageText(
                    locale,
                    `This first plan contains ${formatNumber(collapse.current?.breaks, locale)} breaks and ${formatCurrency(collapse.current?.revenue, locale)} on your channel. There is no frozen baseline to make that zero safe.`,
                    `התוכנית הראשונה הזאת כוללת ${formatNumber(collapse.current?.breaks, locale)} ברייקים ו־${formatCurrency(collapse.current?.revenue, locale)} בערוץ שלכם. אין תוכנית מוקפאת קודמת שהופכת את האפס הזה לבטוח.`,
                  )}
              </p>
              <Button
                className="secondary-button compact"
                type="button"
                variant="outlined"
                disabled={!canEdit || collapseConfirmed}
                onClick={() => setCollapseConfirmed(true)}
              >
                {collapseConfirmed
                  ? pageText(locale, 'Zero-plan freeze enabled', 'הקפאת תוכנית האפס הופעלה')
                  : pageText(locale, 'I understand. Enable this zero-plan freeze', 'הבנתי. אפשר להקפיא את תוכנית האפס')}
              </Button>
            </div>
          )}
          <div className="plan-publish-form">
            <TextField
              size="small"
              label={pageText(locale, 'Name this plan version', 'שם לגרסת התוכנית')}
              value={name}
              onChange={(event) => onNameChange(event.target.value)}
              disabled={!canEdit}
              inputProps={{ maxLength: 120, dir: 'auto' }}
            />
            <TextField
              size="small"
              label={pageText(locale, 'Why, in one line', 'למה, בשורה אחת')}
              value={note}
              onChange={(event) => onNoteChange(event.target.value)}
              disabled={!canEdit}
              inputProps={{ maxLength: 400, dir: 'auto' }}
            />
            <Button
              className="run-button"
              type="button"
              variant="contained"
              disabled={!canEdit || !name.trim() || publishState === 'running' || (collapse.collapsed && !collapseConfirmed)}
              onClick={() => onPublish(collapseConfirmed)}
            >
              <Lock size={15} />
              {publishState === 'running' ? pageText(locale, 'Freezing', 'מקפיא') : words.publish}
            </Button>
          </div>
        </>
      )}

      {alreadyFrozen && (
        <p className="plan-note" role="status">
          {pageText(
            locale,
            'The plan on disk right now is already frozen, byte for byte, as ',
            'התוכנית שעל הדיסק כרגע כבר מוקפאת, בית אחר בית, בשם ',
          )}
          <strong><Name>{frozenVersion?.name || live.frozen_as}</Name></strong>
        </p>
      )}
      {publishError && (
        <p className="plan-note plan-note-red" role="alert"><Name>{publishError}</Name></p>
      )}

      <div className="plan-version-list">
        <div className="plan-section-subhead">
          <h3>{words.planVersion}</h3>
          {/* Linear's device: an open record keeps its place in the set it came
              from, so the whole list can be walked without going back to it. */}
          {selectedIndex >= 0 ? (
            <span className="plan-version-walk">
              <Figure className="numeric">{selectedIndex + 1} / {versions.length}</Figure>
              <button
                type="button"
                aria-label={pageText(locale, 'The version before this one', 'הגרסה שלפני זו')}
                disabled={selectedIndex <= 0}
                onClick={() => walkTo(selectedIndex - 1)}
              >
                <ChevronUp size={14} />
              </button>
              <button
                type="button"
                aria-label={pageText(locale, 'The version after this one', 'הגרסה שאחרי זו')}
                disabled={selectedIndex >= versions.length - 1}
                onClick={() => walkTo(selectedIndex + 1)}
              >
                <ChevronDown size={14} />
              </button>
            </span>
          ) : (
            <span>{formatNumber(versions.length, locale)}</span>
          )}
        </div>
        {versions.length === 0 ? (
          <p className="plan-note">
            {pageText(
              locale,
              'No plan has been frozen yet. The first freeze becomes the version everyone downstream reads by name.',
              'עדיין לא הוקפאה אף תוכנית. ההקפאה הראשונה תהפוך לגרסה שכולם בהמשך קוראים בשמה.',
            )}
          </p>
        ) : (
          versions.map((version) => (
            <VersionRow
              key={version.version_id}
              version={version}
              locale={locale}
              selected={version.version_id === selectedId}
              canEdit={canEdit}
              onSelect={onSelect}
              onDiff={onDiff}
              onRestore={onRestore}
            />
          ))
        )}
      </div>

      {diff && diff.available && (
        <div className="plan-diff">
          <div className="plan-section-subhead">
            <h3>
              {pageText(locale, 'What changed', 'מה השתנה')}
              {' '}<Name>{diff.version_name}</Name>{' '}
              {pageText(locale, 'against', 'מול')}
              {' '}<Name>{diff.against_name || diff.against}</Name>
            </h3>
            {scopeLine(diff.scope, locale) ? <span>{scopeLine(diff.scope, locale)}</span> : null}
          </div>
          {diff.identical ? (
            <p className="plan-note">
              {pageText(locale, 'Nothing moved between these two.', 'שום דבר לא זז בין השתיים.')}
            </p>
          ) : (
            <>
              <div className="plan-figure-row">
                <div className="plan-figure">
                  <span>{words.expectedRevenue}</span>
                  <strong><Numeric>{formatCurrency(diff.delta?.revenue, locale)}</Numeric></strong>
                </div>
                <div className="plan-figure">
                  <span>{words.breaks}</span>
                  <strong><Numeric>{formatNumber(diff.delta?.breaks, locale)}</Numeric></strong>
                </div>
                <div className="plan-figure">
                  <span>{pageText(locale, 'Plan rows', 'שורות תוכנית')}</span>
                  <strong><Numeric>{formatNumber(diff.delta?.rows, locale)}</Numeric></strong>
                </div>
                <div className="plan-figure">
                  <span>{pageText(locale, 'Days that moved', 'ימים שזזו')}</span>
                  <strong><Numeric>{formatNumber(diff.changed_days?.length || 0, locale)}</Numeric></strong>
                </div>
              </div>
              <ul className="plan-diff-days">
                {(diff.changed_days || []).slice(0, 12).map((day) => (
                  <li key={day.date}>
                    <Numeric>{day.date}</Numeric>
                    <span>
                      <Numeric>{formatNumber(day.breaks_delta, locale)}</Numeric> {words.breaks}
                    </span>
                    <strong><Numeric>{formatCurrency(day.revenue_delta, locale)}</Numeric></strong>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
      {diff && diff.available === false && (
        <p className="plan-note plan-note-amber" role="status">{diffReason(diff, locale)}</p>
      )}
    </section>
  );
}

export default PublishPanel;
