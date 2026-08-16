import { SECTIONS, sectionLabel } from './plan-week-model';

// Every command on Plan, the week, in one list.
//
// One list serves both the palette and the keyboard, so a shortcut printed in
// the palette is the shortcut that fires, and a command that cannot run right
// now is refused in both places with the same reason rather than being hidden in
// one and dead in the other.
//
// Labels are functions of the locale rather than strings, so the list can be
// built once and read in either language.

export function planCommands({
  go, words, surface, boardView, setBoardView, adoptLeg, runNow, compareNow,
  optimizationAllowed, optimizationBlockedReason, requestPublish, openPalette,
}) {
  const navigation = SECTIONS.map((section) => ({
    id: `go-${section.id}`,
    group: (locale) => (locale === 'he' ? 'מעבר' : 'Go to'),
    label: (locale) => sectionLabel(section.id, locale),
    keywords: [section.id, section.key],
    shortcut: ['g', section.key],
    run: () => go(section.id),
  }));

  const zooms = ['grid', 'strip', 'timeline', 'day'].map((view, index) => ({
    id: `board-${view}`,
    group: (locale) => (locale === 'he' ? 'לוח השבוע' : 'Week board'),
    label: (locale) => {
      const names = {
        grid: locale === 'he' ? 'רשת' : 'Grid',
        strip: locale === 'he' ? 'רצועות שידור' : 'Broadcast strips',
        timeline: locale === 'he' ? 'ציר זמן' : 'Timeline',
        day: locale === 'he' ? 'יום אחד, לעריכה' : 'One day, editable',
      };
      return names[view];
    },
    keywords: [view, 'zoom', 'board'],
    shortcut: [String(index + 1)],
    run: () => {
      setBoardView(view);
      go('board');
    },
  }));

  const actions = [
    {
      id: 'save-objective',
      group: (locale) => (locale === 'he' ? 'פעולות' : 'Actions'),
      label: (locale) => (locale === 'he' ? 'שמירת המטרה' : 'Save the objective'),
      keywords: ['save', 'objective', 'שמירה'],
      shortcut: ['mod', 's'],
      disabled: !surface.dirty || surface.saveState === 'saving',
      disabledReason: (locale) => (locale === 'he' ? 'אין שינוי לשמור' : 'nothing to save'),
      run: () => { go('objective'); surface.saveObjective(); },
    },
    {
      id: 'run-plan',
      group: (locale) => (locale === 'he' ? 'פעולות' : 'Actions'),
      label: () => words.run,
      keywords: ['run', 'plan', 'הרצה'],
      shortcut: ['r'],
      disabled: surface.runState === 'running' || !optimizationAllowed,
      disabledReason: (locale) => (surface.runState === 'running'
        ? (locale === 'he' ? 'הרצה כבר פועלת' : 'a run is already going')
        : optimizationBlockedReason),
      // The same function the state row's control calls, so the palette row and
      // the button on the header cannot drift into doing different things.
      run: runNow,
    },
    {
      id: 'compare',
      group: (locale) => (locale === 'he' ? 'פעולות' : 'Actions'),
      label: (locale) => (locale === 'he' ? 'השוואת שני תרחישים' : 'Compare two scenarios'),
      keywords: ['compare', 'scenario', 'net', 'השוואה'],
      shortcut: ['c'],
      disabled: surface.compareState === 'running' || !optimizationAllowed,
      disabledReason: (locale) => (surface.compareState === 'running'
        ? (locale === 'he' ? 'השוואה כבר פועלת' : 'a comparison is already going')
        : optimizationBlockedReason),
      run: compareNow,
    },
    {
      // The comparison is fourteen real optimizations and takes about eleven
      // seconds, so stopping it is an act of its own and belongs on the keyboard
      // beside the act that starts it.
      id: 'stop-compare',
      group: (locale) => (locale === 'he' ? 'פעולות' : 'Actions'),
      label: (locale) => (locale === 'he' ? 'עצירת ההשוואה' : 'Stop the comparison'),
      keywords: ['stop', 'cancel', 'compare', 'עצירה', 'השוואה'],
      shortcut: ['mod', '.'],
      disabled: surface.compareState !== 'running',
      disabledReason: (locale) => (locale === 'he' ? 'אין השוואה שרצה' : 'no comparison is running'),
      run: () => { surface.cancelCompare(); },
    },
    // Adopting the leg that won the comparison is the act step 3 exists for, so
    // it is a command like any other: same chord lead for both legs, refused
    // with its reason until a comparison has actually finished and reported the
    // levers it ran under.
    ...['a', 'b'].map((leg) => ({
      id: `adopt-${leg}`,
      group: (locale) => (locale === 'he' ? 'פעולות' : 'Actions'),
      label: (locale) => (locale === 'he'
        ? `קביעת תרחיש ${leg.toUpperCase()} כמטרה`
        : `Use scenario ${leg.toUpperCase()} as the objective`),
      keywords: ['adopt', 'use', 'scenario', 'objective', 'winner', leg, 'מטרה', 'תרחיש'],
      shortcut: ['u', leg],
      disabled: !surface.adoptable(leg),
      disabledReason: (locale) => (locale === 'he'
        ? 'אין השוואה שהסתיימה ודיווחה את הידיות שלה'
        : 'no finished comparison has reported its levers'),
      run: () => { adoptLeg(leg); },
    })),
    {
      id: 'publish',
      group: (locale) => (locale === 'he' ? 'פעולות' : 'Actions'),
      label: () => words.publish,
      keywords: ['publish', 'freeze', 'version', 'הפצה', 'הקפאה'],
      shortcut: ['p'],
      disabled: !surface.canPublish || !surface.versionName.trim(),
      disabledReason: (locale) => (surface.canPublish
        ? (locale === 'he' ? 'תנו שם לגרסה קודם' : 'name the version first')
        : (surface.canPublishReason || (locale === 'he' ? 'אין הרשאה' : 'not permitted'))),
      // Publishing is a write. The palette and P shortcut therefore enter the
      // same scope-and-consequence review as the visible publish control.
      run: () => { go('publish'); requestPublish(false); },
    },
  ];

  // The revenue and yield owner's question, reachable by name from anywhere on
  // the destination rather than only by finding the panel it sits in.
  const questions = [
    {
      id: 'worth-of-a-second',
      group: (locale) => (locale === 'he' ? 'שאלות' : 'Questions'),
      label: (locale) => (locale === 'he' ? 'כמה שווה שנייה של זמן שידור' : 'What a second of airtime is worth'),
      keywords: ['yield', 'second', 'worth', 'rate', 'שווי', 'שנייה', 'תעריף'],
      shortcut: ['g', 'y'],
      run: () => go('supply'),
    },
  ];

  // The palette's own row sits with the other actions rather than after the
  // zooms, because the list is grouped by contiguous runs and a group that
  // appears twice reads as two different groups with one name.
  actions.push({
    id: 'palette',
    group: (locale) => (locale === 'he' ? 'פעולות' : 'Actions'),
    label: (locale) => (locale === 'he' ? 'פתיחת לוח הפקודות' : 'Open the command palette'),
    keywords: ['palette', 'command', 'פקודות'],
    shortcut: ['mod', 'k'],
    run: openPalette,
  });

  return [...actions, ...navigation, ...questions, ...zooms].map((command) => ({
    ...command,
    disabled: Boolean(command.disabled),
    _boardView: boardView,
  }));
}

export default planCommands;
