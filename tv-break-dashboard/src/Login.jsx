import React, { useState } from 'react';
import './login.css';

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

export function fetchMe() {
  return authRequest('/api/auth/me');
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

function loginErrorText(result) {
  if (result.status === 401) return 'שם המשתמש או הסיסמה שגויים.';
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
export default function Login({ onLoggedIn }) {
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
    <div className="login-screen" dir="rtl" lang="he">
      <form className="login-card" onSubmit={submit}>
        <div className="login-brand">
          <div className="login-brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <strong>Kairos</strong>
            <small>ניהול הכנסות מפרסום</small>
          </div>
        </div>
        <h1>כניסה למערכת</h1>
        <p className="login-sub">יש להיכנס עם החשבון האישי שהוקצה לך.</p>
        <label className="login-field">
          <span>שם משתמש</span>
          <input
            dir="ltr"
            autoComplete="username"
            autoFocus
            value={username}
            onChange={(event) => setUsername(event.target.value)}
          />
        </label>
        <label className="login-field">
          <span>סיסמה</span>
          <input
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
        <button className="login-submit" type="submit" disabled={busy || username.trim() === '' || password === ''}>
          {busy ? 'רק רגע...' : 'כניסה'}
        </button>
      </form>
    </div>
  );
}

// Change-password dialog. In forced mode (must_change_password) there is no
// way to dismiss it: the temporary password has to be replaced first.
export function ChangePasswordDialog({ locale = 'he', forced = false, onClose, onDone }) {
  const t = (en, he) => (locale === 'he' ? he : en);
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
    <div className="auth-overlay" dir={locale === 'he' ? 'rtl' : 'ltr'} role="dialog" aria-modal="true">
      <form className="auth-dialog" onSubmit={submit}>
        {forced ? null : (
          <button type="button" className="auth-close" onClick={onClose} aria-label={t('Close', 'סגירה')}>
            ×
          </button>
        )}
        <h2>{t('Change password', 'החלפת סיסמה')}</h2>
        {forced && (
          <p className="auth-note">
            {t(
              'The temporary password must be replaced before continuing to the workspace.',
              'נדרש להחליף את הסיסמה הזמנית לפני המשך העבודה במערכת.',
            )}
          </p>
        )}
        <label className="auth-field">
          <span>{t('Current password', 'סיסמה נוכחית')}</span>
          <input
            type="password"
            dir="ltr"
            autoComplete="current-password"
            autoFocus
            value={current}
            onChange={(event) => setCurrent(event.target.value)}
          />
        </label>
        <label className="auth-field">
          <span>{t('New password', 'סיסמה חדשה')}</span>
          <input
            type="password"
            dir="ltr"
            autoComplete="new-password"
            value={next}
            onChange={(event) => setNext(event.target.value)}
          />
        </label>
        <label className="auth-field">
          <span>{t('Confirm the new password', 'אימות הסיסמה החדשה')}</span>
          <input
            type="password"
            dir="ltr"
            autoComplete="new-password"
            value={confirm}
            onChange={(event) => setConfirm(event.target.value)}
          />
        </label>
        <p className="auth-hint">
          {t(
            `At least ${MIN_PASSWORD_LENGTH} characters.`,
            `לפחות ${MIN_PASSWORD_LENGTH} תווים.`,
          )}
        </p>
        {error && (
          <p className="auth-error" role="alert">
            {error}
          </p>
        )}
        <div className="auth-actions">
          {forced ? null : (
            <button type="button" className="auth-secondary" onClick={onClose} disabled={busy}>
              {t('Cancel', 'ביטול')}
            </button>
          )}
          <button
            type="submit"
            className="auth-primary"
            disabled={busy || current === '' || next === '' || confirm === ''}
          >
            {busy ? t('Just a moment...', 'רק רגע...') : t('Update password', 'עדכון הסיסמה')}
          </button>
        </div>
      </form>
    </div>
  );
}
