import React, { useRef, useState } from 'react';
import './login.css';
import { DirectionRoot } from './bidi';
import { KairosMark } from './kairos-icons';
import { Button } from '../studio/actions';
import { Card } from './primitives';
import { InputControl } from './dom-controls';
import { Dialog } from '../studio/modal';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

// All auth calls carry credentials so the kairos_session cookie flows on the
// same-origin production setup (dist served by the API) and on a same-site
// dev origin. Non-2xx and network failures are returned, never thrown, so
// every caller renders an honest error state.
async function authRequest(path, options = {}) {
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      credentials: 'include',
      headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
      ...options,
    });
    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }
    return {
      ok: response.ok,
      status: response.status,
      data,
      retryAfter: Number(response.headers.get('Retry-After') || 0),
    };
  } catch {
    return { ok: false, status: 0, data: null, retryAfter: 0 };
  }
}

let sessionProbe = null;
let sessionProbeIssue = '';

export function setSessionProbeIssue(value) {
  sessionProbeIssue = value === 'offline' || value === 'setup' ? value : '';
}

export function fetchMe() {
  if (!sessionProbe) {
    sessionProbe = authRequest('/api/auth/session').finally(() => {
      sessionProbe = null;
    });
  }
  return sessionProbe;
}

export function requestLogin(username, password) {
  return authRequest('/api/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) });
}

export function requestLogout() {
  return authRequest('/api/auth/logout', { method: 'POST' });
}

export function requestPasswordChange(currentPassword, newPassword) {
  return authRequest('/api/auth/change-password', {
    method: 'POST',
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
}

export function fetchAccounts() {
  return authRequest('/api/auth/users');
}

export function createAccount(payload) {
  return authRequest('/api/auth/users', { method: 'POST', body: JSON.stringify(payload) });
}

export function deleteAccount(username) {
  return authRequest(`/api/auth/users/${encodeURIComponent(username)}`, { method: 'DELETE' });
}

export function resetAccountPassword(username, newPassword) {
  return authRequest(`/api/auth/users/${encodeURIComponent(username)}/reset-password`, {
    method: 'POST',
    body: JSON.stringify({ new_password: newPassword }),
  });
}

export function setAccountAffiliation(username, affiliation) {
  return authRequest(`/api/auth/users/${encodeURIComponent(username)}/affiliation`, {
    method: 'PUT',
    body: JSON.stringify({ affiliation }),
  });
}

export const MIN_PASSWORD_LENGTH = 10;

const ROLE_LABELS = {
  admin: { en: 'Admin', he: 'ניהול' },
  operator: { en: 'Operator', he: 'תפעול' },
  viewer: { en: 'Viewer', he: 'צפייה' },
};

export function roleLabel(role, locale) {
  const labels = ROLE_LABELS[role];
  if (!labels) return role || '';
  return locale === 'he' ? labels.he : labels.en;
}

const AFFILIATION_LABELS = {
  company: { en: 'Company', he: 'חברה' },
  channel: { en: 'Channel', he: 'ערוץ' },
  unknown: { en: 'Unresolved', he: 'לא הוגדר' },
};

export function affiliationLabel(affiliation, locale) {
  const labels = AFFILIATION_LABELS[affiliation] || AFFILIATION_LABELS.unknown;
  return locale === 'he' ? labels.he : labels.en;
}

function loginErrorText(result) {
  if (result.status === 401) return 'שם המפעיל או הסיסמה שגויים.';
  if (result.status === 429) {
    const minutes = Math.max(1, Math.ceil((result.retryAfter || 60) / 60));
    return `יותר מדי ניסיונות כניסה. אפשר לנסות שוב בעוד ${minutes} דקות בערך.`;
  }
  if (result.status === 503) {
    return 'הכניסה עדיין לא הוגדרה בשרת. יש להריץ את scripts/init_auth.py ולנסות שוב.';
  }
  if (result.status === 0) return 'אין חיבור לשרת. יש לוודא שהשרת פועל ולנסות שוב.';
  return `הכניסה נכשלה (סטטוס ${result.status}).`;
}

// Full-screen sign-in card. Hebrew-first by design: it renders before the
// operator settings (and their language choice) are available.
function SessionUnavailable({ issue }) {
  const setupRequired = issue === 'setup';
  return (
    <DirectionRoot locale="he" as="main" className="login-screen" lang="he" aria-labelledby="session-unavailable-title">
      <Card className="auth-session-unavailable" role="alert" aria-describedby="session-unavailable-detail">
        <KairosMark size={44} title="Kairos" />
        <p className="login-kicker">{setupRequired ? 'נדרשת הגדרת גישה' : 'הגישה מושהית'}</p>
        <h1 id="session-unavailable-title">
          {setupRequired ? 'הכניסה למערכת עדיין לא הוגדרה.' : 'לא ניתן לאמת את הגישה כרגע.'}
        </h1>
        <p id="session-unavailable-detail" className="login-sub">
          {setupRequired
            ? 'יש להשלים את הגדרת האימות בשרת באמצעות scripts/init_auth.py, ואז לנסות שוב.'
            : 'השרת אינו זמין, ולכן סביבת העבודה נשארת נעולה. יש לבדוק את החיבור ולנסות שוב.'}
        </p>
        <Button className="login-submit" type="button" variant="contained" onClick={() => window.location.reload()}>
          ניסיון נוסף
        </Button>
      </Card>
    </DirectionRoot>
  );
}

function LoginForm({ onLoggedIn }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function submit(event) {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError('');
    const result = await requestLogin(username.trim().toLowerCase(), password);
    if (result.ok && result.data && result.data.username) {
      onLoggedIn(result.data);
      return;
    }
    setBusy(false);
    setError(loginErrorText(result));
  }

  return (
    <DirectionRoot locale="he" className="login-screen" lang="he">
      <Card as="div" className="login-stage">
        <section className="login-context" aria-label="על סביבת העבודה">
          <div className="login-brand">
            <KairosMark size={44} title="Kairos" />
            <div>
              <strong>KAIROS</strong>
              <small>מערכת תפעול שידור</small>
            </div>
          </div>
          <div className="login-context-copy">
            <p className="login-kicker">סביבת עבודה מבוקרת</p>
            <h2>תכנון, מלאי, מסחרי וממשל במקום אחד.</h2>
            <p>הגישה אישית ומוגבלת לפי התפקיד. הפעולות מתבצעות מתוך ההקשר התפעולי שבו הן חלות.</p>
          </div>
          <p className="login-context-foot">תכנון שידור · מלאי · מסחרי · ממשל</p>
        </section>

        <form className="login-card" onSubmit={submit} aria-labelledby="login-title">
          <div className="login-form-head">
            <span>גישה למערכת</span>
            <h1 id="login-title">כניסה ל־Kairos</h1>
            <p className="login-sub">השתמשו בחשבון האישי שהוקצה לכם.</p>
          </div>
          <label className="login-field">
            <span>שם משתמש</span>
            <InputControl
              dir="ltr"
              autoComplete="username"
              autoFocus
              value={username}
              onChange={(event) => setUsername(event.target.value)}
            />
          </label>
          <label className="login-field">
            <span>סיסמה</span>
            <InputControl
              type="password"
              dir="ltr"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
          {error && (
            <p className="login-error" role="alert">
              {error}
            </p>
          )}
          <Button className="login-submit" type="submit" variant="contained" loading={busy} disabled={username.trim() === '' || password === ''}>
            {busy ? 'מתחבר...' : 'כניסה למערכת'}
          </Button>
          <p className="login-account-note">החשבון וההרשאות מנוהלים על ידי מנהל המערכת.</p>
        </form>
      </Card>
    </DirectionRoot>
  );
}

export default function Login(props) {
  return sessionProbeIssue ? <SessionUnavailable issue={sessionProbeIssue} /> : <LoginForm {...props} />;
}

// Change-password dialog. In forced mode (must_change_password) there is no
// way to dismiss it: the temporary password has to be replaced first.
export function ChangePasswordDialog({ locale = 'he', forced = false, onClose, onDone }) {
  const t = (en, he) => (locale === 'he' ? he : en);
  const currentRef = useRef(null);
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [confirm, setConfirm] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function submit(event) {
    event.preventDefault();
    if (busy) return;
    if (next.length < MIN_PASSWORD_LENGTH) {
      setError(t(
        `The new password must be at least ${MIN_PASSWORD_LENGTH} characters long.`,
        `הסיסמה החדשה צריכה להיות באורך ${MIN_PASSWORD_LENGTH} תווים לפחות.`,
      ));
      return;
    }
    if (next !== confirm) {
      setError(t('The password confirmation does not match.', 'אימות הסיסמה אינו תואם לסיסמה החדשה.'));
      return;
    }
    setBusy(true);
    setError('');
    const result = await requestPasswordChange(current, next);
    if (result.ok && result.data) {
      onDone(result.data);
      return;
    }
    setBusy(false);
    if (result.status === 403) setError(t('The current password is incorrect.', 'הסיסמה הנוכחית שגויה.'));
    else if (result.status === 0) setError(t('No connection to the server.', 'אין חיבור לשרת. יש לנסות שוב.'));
    else setError(t(`The password change failed (status ${result.status}).`, `החלפת הסיסמה נכשלה (סטטוס ${result.status}).`));
  }

  return (
    <DirectionRoot locale={locale}>
      <Dialog
        open
        size="narrow"
        className={forced ? 'auth-password-dialog auth-dialog-forced' : 'auth-password-dialog'}
        title={t('Change password', 'החלפת סיסמה')}
        description={forced
          ? t(
              'The temporary password must be replaced before continuing to the workspace.',
              'נדרש להחליף את הסיסמה הזמנית לפני המשך העבודה במערכת.',
            )
          : t('Choose a new password for this account.', 'בחרו סיסמה חדשה לחשבון הזה.')}
        closeLabel={t('Close', 'סגירה')}
        initialFocusRef={currentRef}
        dismissOnBackdrop={!forced}
        onClose={forced ? undefined : onClose}
      >
        <form
          className="auth-password-form"
          onSubmit={submit}
          onKeyDown={forced ? (event) => {
            if (event.key === 'Escape') event.preventDefault();
          } : undefined}
        >
          <label className="auth-field">
            <span>{t('Current password', 'סיסמה נוכחית')}</span>
            <InputControl
              ref={currentRef}
              type="password"
              dir="ltr"
              autoComplete="current-password"
              value={current}
              onChange={(event) => setCurrent(event.target.value)}
            />
          </label>
          <label className="auth-field">
            <span>{t('New password', 'סיסמה חדשה')}</span>
            <InputControl
              type="password"
              dir="ltr"
              autoComplete="new-password"
              value={next}
              onChange={(event) => setNext(event.target.value)}
            />
          </label>
          <label className="auth-field">
            <span>{t('Confirm the new password', 'אימות הסיסמה החדשה')}</span>
            <InputControl
              type="password"
              dir="ltr"
              autoComplete="new-password"
              value={confirm}
              onChange={(event) => setConfirm(event.target.value)}
            />
          </label>
          <p className="auth-hint">
            {t(`At least ${MIN_PASSWORD_LENGTH} characters.`, `לפחות ${MIN_PASSWORD_LENGTH} תווים.`)}
          </p>
          {error && <p className="auth-error" role="alert">{error}</p>}
          <div className="auth-actions">
            {forced ? null : (
              <Button type="button" variant="outlined" className="auth-secondary" onClick={onClose} disabled={busy}>
                {t('Cancel', 'ביטול')}
              </Button>
            )}
            <Button type="submit" variant="contained" className="auth-primary" loading={busy} disabled={current === '' || next === '' || confirm === ''}>
              {busy ? t('Just a moment...', 'רק רגע...') : t('Update password', 'עדכון הסיסמה')}
            </Button>
          </div>
        </form>
      </Dialog>
    </DirectionRoot>
  );
}
