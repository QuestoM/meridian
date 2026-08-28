import React from 'react';
import { Button } from '../studio/actions';
import { pageText } from './format';
import './shell-error.css';

// A blank page is the worst thing this product can show, because it is the one
// state that says nothing at all: not what failed, not whether the data is
// safe, not what to do. Without a boundary, React unmounts the whole tree on
// any render error - the workspace AND the navigation rail - and the operator
// is left with white and a support call.
//
// Measured on a first-run study: a deploy landed while the browser held the
// previous build, the next navigation asked for a code chunk that no longer
// existed, and the app vanished. That is not an exotic failure; it is what a
// deploy DOES to every session already open, and this product deploys while
// people are working in it.
//
// So the boundary separates the two cases it can honestly tell apart. A chunk
// that will not load means the running page is older than the server: the data
// is untouched and a reload fixes it, and the message says exactly that. Any
// other error is a real fault: the boundary says so plainly, states that
// nothing was lost from the saved data, and still offers the reload rather
// than leaving the person stranded.

const STALE_BUILD_SIGNS = [
  'failed to fetch dynamically imported module',
  'error loading dynamically imported module',
  'loading chunk',
  'importing a module script failed',
  'unable to preload',
];

function isStaleBuild(error) {
  const message = String((error && error.message) || error || '').toLowerCase();
  return STALE_BUILD_SIGNS.some((sign) => message.includes(sign));
}

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // The console is where a developer looks; the screen is where the operator
    // looks. Both get told, and neither is left guessing.
    // eslint-disable-next-line no-console
    console.error('Kairos surface error', error, info);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    const locale = this.props.locale === 'en' ? 'en' : 'he';
    const stale = isStaleBuild(error);
    return (
      <div className="shell-error" role="alert" dir={locale === 'he' ? 'rtl' : 'ltr'}>
        <h1>
          {stale
            ? pageText(locale, 'A newer version was released', 'שוחררה גרסה חדשה יותר')
            : pageText(locale, 'This screen could not be drawn', 'לא ניתן היה להציג את המסך הזה')}
        </h1>
        <p>
          {stale
            ? pageText(
              locale,
              'The page open here is older than the one on the server, so part of it could no longer be loaded. Nothing was saved or lost; reloading brings up the current version.',
              'העמוד שפתוח כאן ישן מזה שעל השרת, ולכן חלק ממנו כבר לא ניתן לטעינה. שום דבר לא נשמר ולא אבד; רענון יביא את הגרסה הנוכחית.',
            )
            : pageText(
              locale,
              'Something in this screen failed while rendering. Your saved data was not touched by this failure. Reloading usually restores the screen; if it does not, the details are in the browser console.',
              'משהו במסך הזה נכשל בזמן ההצגה. הנתונים השמורים שלכם לא נגעו בכשל הזה. רענון בדרך כלל מחזיר את המסך; אם לא, הפרטים נמצאים בקונסולת הדפדפן.',
            )}
        </p>
        <Button type="button" variant="contained" onClick={() => window.location.reload()}>
          {pageText(locale, 'Reload', 'רענון')}
        </Button>
      </div>
    );
  }
}
