import React, { useEffect, useState } from 'react';
import { Tooltip } from '@mui/material';
import { pageText } from './format';
import {
  MIN_PASSWORD_LENGTH,
  affiliationLabel,
  createAccount,
  deleteAccount,
  fetchAccounts,
  resetAccountPassword,
  roleLabel,
  setAccountAffiliation,
} from './Login';

// Admin-only account management over /api/auth/users*: list, create, delete
// and reset passwords. Every failure surfaces honestly; nothing is optimistic.
export function UserAdminDialog({ locale, selfUsername, notify, onClose }) {
  const t = (en, he) => pageText(locale, en, he);
  const [accounts, setAccounts] = useState([]);
  const [loadState, setLoadState] = useState('loading');
  const [reloadKey, setReloadKey] = useState(0);
  const [form, setForm] = useState({ username: '', display_name: '', role: 'viewer', affiliation: 'company', password: '' });
  const [formError, setFormError] = useState('');
  const [busy, setBusy] = useState(false);
  const [resetFor, setResetFor] = useState('');
  const [resetValue, setResetValue] = useState('');
  const [rowError, setRowError] = useState('');
  const [confirmDelete, setConfirmDelete] = useState('');

  useEffect(() => {
    let active = true;
    setLoadState('loading');
    fetchAccounts().then((result) => {
      if (!active) return;
      if (result.ok && result.data && Array.isArray(result.data.users)) {
        setAccounts(result.data.users);
        setLoadState('ready');
      } else {
        setLoadState('error');
      }
    });
    return () => {
      active = false;
    };
  }, [reloadKey]);

  const adminTotal = accounts.filter((account) => account.role === 'admin').length;

  function describeFailure(result) {
    if (result.status === 0) return t('No connection to the server.', 'אין חיבור לשרת.');
    if (result.status === 409) return t('That username is already taken.', 'שם המשתמש הזה כבר תפוס.');
    if (result.status === 422) {
      return t(
        `The password needs at least ${MIN_PASSWORD_LENGTH} characters.`,
        `הסיסמה צריכה להכיל לפחות ${MIN_PASSWORD_LENGTH} תווים.`,
      );
    }
    const detail = result.data && result.data.detail ? String(result.data.detail) : '';
    if (detail.toLowerCase().includes('last admin')) {
      return t('The last admin account cannot be deleted.', 'אי אפשר למחוק את חשבון הניהול האחרון.');
    }
    if (detail.toLowerCase().includes('signed in with')) {
      return t('You cannot delete the account you are signed in with.', 'אי אפשר למחוק את החשבון שאיתו נכנסת למערכת.');
    }
    if (detail) return detail;
    return t(`The request failed (status ${result.status}).`, `הפעולה נכשלה (סטטוס ${result.status}).`);
  }

  async function submitCreate(event) {
    event.preventDefault();
    if (busy) return;
    if (form.password.length < MIN_PASSWORD_LENGTH) {
      setFormError(t(
        `The temporary password needs at least ${MIN_PASSWORD_LENGTH} characters.`,
        `הסיסמה הזמנית צריכה להכיל לפחות ${MIN_PASSWORD_LENGTH} תווים.`,
      ));
      return;
    }
    setBusy(true);
    setFormError('');
    const result = await createAccount({
      username: form.username.trim().toLowerCase(),
      password: form.password,
      role: form.role,
      display_name: form.display_name.trim(),
      must_change_password: true,
      affiliation: form.affiliation,
    });
    setBusy(false);
    if (result.ok && result.data) {
      setForm({ username: '', display_name: '', role: 'viewer', affiliation: 'company', password: '' });
      setReloadKey((key) => key + 1);
      notify('Account created.', 'החשבון נוצר.');
    } else {
      setFormError(describeFailure(result));
    }
  }

  async function submitReset(username) {
    if (busy) return;
    setBusy(true);
    setRowError('');
    const result = await resetAccountPassword(username, resetValue);
    setBusy(false);
    if (result.ok) {
      setResetFor('');
      setResetValue('');
      setReloadKey((key) => key + 1);
      notify('Temporary password set.', 'נקבעה סיסמה זמנית חדשה.');
    } else {
      setRowError(describeFailure(result));
    }
  }

  async function submitAffiliation(username, affiliation) {
    if (busy) return;
    setBusy(true);
    setRowError('');
    const result = await setAccountAffiliation(username, affiliation);
    setBusy(false);
    if (result.ok) {
      setReloadKey((key) => key + 1);
      notify('Affiliation updated.', 'השיוך עודכן.');
    } else {
      setRowError(describeFailure(result));
    }
  }

  async function submitDelete(username) {
    if (busy) return;
    if (confirmDelete !== username) {
      setConfirmDelete(username);
      return;
    }
    setBusy(true);
    setRowError('');
    const result = await deleteAccount(username);
    setBusy(false);
    setConfirmDelete('');
    if (result.ok) {
      setReloadKey((key) => key + 1);
      notify('Account deleted.', 'החשבון נמחק.');
    } else {
      setRowError(describeFailure(result));
    }
  }

  return (
    <div className="auth-overlay" dir={locale === 'he' ? 'rtl' : 'ltr'} role="dialog" aria-modal="true">
      <div className="auth-dialog auth-dialog-wide">
        <button type="button" className="auth-close" onClick={onClose} aria-label={t('Close', 'סגירה')}>
          ×
        </button>
        <h2>{t('Manage accounts', 'ניהול חשבונות')}</h2>
        <p className="auth-hint">
          {t(
            'Each teammate signs in with a personal account; the role decides what the account can change.',
            'לכל אחד ואחת בצוות חשבון אישי; התפקיד קובע אילו פעולות פתוחות בחשבון.',
          )}
        </p>
        {loadState === 'loading' && <p className="auth-empty">{t('Loading accounts...', 'רק רגע...')}</p>}
        {loadState === 'error' && (
          <div>
            <p className="auth-error">{t('Could not load the account list.', 'טעינת רשימת החשבונות נכשלה.')}</p>
            <div className="auth-actions">
              <button type="button" className="auth-secondary" onClick={() => setReloadKey((key) => key + 1)}>
                {t('Try again', 'ניסיון נוסף')}
              </button>
            </div>
          </div>
        )}
        {loadState === 'ready' && (
          <table className="auth-table">
            <thead>
              <tr>
                <th>{t('Name', 'שם')}</th>
                <th>{t('Display name', 'שם תצוגה')}</th>
                <th>{t('Role', 'תפקיד')}</th>
                <th>{t('Affiliation', 'שיוך')}</th>
                <th aria-label={t('Actions', 'פעולות')} />
              </tr>
            </thead>
            <tbody>
              {accounts.map((account) => {
                const isSelf = account.username === selfUsername;
                const lastAdmin = account.role === 'admin' && adminTotal <= 1;
                return (
                  <React.Fragment key={account.username}>
                    <tr>
                      <td className="auth-mono">{account.username}</td>
                      <td>{account.display_name}</td>
                      <td>
                        {roleLabel(account.role, locale)}
                        {account.must_change_password && (
                          <span className="auth-flag">{t('Temporary password', 'סיסמה זמנית')}</span>
                        )}
                      </td>
                      <td>
                        <select
                          value={account.affiliation === 'channel' ? 'channel' : 'company'}
                          disabled={busy}
                          aria-label={t('Affiliation', 'שיוך')}
                          onChange={(event) => submitAffiliation(account.username, event.target.value)}
                        >
                          <option value="company">{affiliationLabel('company', locale)}</option>
                          <option value="channel">{affiliationLabel('channel', locale)}</option>
                        </select>
                      </td>
                      <td>
                        <div className="auth-row-actions">
                          <button
                            type="button"
                            className="auth-mini"
                            disabled={busy}
                            onClick={() => {
                              setResetFor(resetFor === account.username ? '' : account.username);
                              setResetValue('');
                              setRowError('');
                            }}
                          >
                            {t('Reset password', 'איפוס סיסמה')}
                          </button>
                          <Tooltip
                            title={
                              isSelf
                                ? t(
                                    'You cannot delete the account you are signed in with.',
                                    'אי אפשר למחוק את החשבון שאיתו נכנסת למערכת.',
                                  )
                                : lastAdmin
                                  ? t('The last admin account cannot be deleted.', 'אי אפשר למחוק את חשבון הניהול האחרון.')
                                  : ''
                            }
                            arrow
                            placement="bottom"
                          >
                            <span className="auth-mini-wrap">
                              <button
                                type="button"
                                className={`auth-mini auth-danger${confirmDelete === account.username ? ' auth-confirming' : ''}`}
                                disabled={busy || isSelf || lastAdmin}
                                onClick={() => submitDelete(account.username)}
                              >
                                {confirmDelete === account.username ? t('Confirm delete', 'אישור מחיקה') : t('Delete', 'מחיקה')}
                              </button>
                            </span>
                          </Tooltip>
                        </div>
                      </td>
                    </tr>
                    {resetFor === account.username && (
                      <tr className="auth-reset-row">
                        <td colSpan={5}>
                          <div className="auth-inline-form">
                            <input
                              type="password"
                              dir="ltr"
                              autoComplete="new-password"
                              placeholder={t('New temporary password', 'סיסמה זמנית חדשה')}
                              value={resetValue}
                              onChange={(event) => setResetValue(event.target.value)}
                            />
                            <button
                              type="button"
                              className="auth-mini"
                              disabled={busy || resetValue.length < MIN_PASSWORD_LENGTH}
                              onClick={() => submitReset(account.username)}
                            >
                              {t('Set password', 'קביעת הסיסמה')}
                            </button>
                            <span className="auth-hint">
                              {t('A change is required at the next sign-in.', 'בכניסה הבאה תידרש החלפת סיסמה.')}
                            </span>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        )}
        {loadState === 'ready' && accounts.length === 0 && (
          <p className="auth-empty">{t('No accounts yet.', 'אין עדיין חשבונות.')}</p>
        )}
        {loadState === 'ready' && rowError && (
          <p className="auth-error" role="alert">
            {rowError}
          </p>
        )}
        {loadState === 'ready' && (
          <form onSubmit={submitCreate}>
            <h3>{t('New account', 'חשבון חדש')}</h3>
            <div className="auth-create-grid">
              <label className="auth-field">
                <span>{t('Username', 'שם משתמש')}</span>
                <input
                  dir="ltr"
                  autoComplete="off"
                  value={form.username}
                  onChange={(event) => setForm({ ...form, username: event.target.value })}
                />
              </label>
              <label className="auth-field">
                <span>{t('Display name', 'שם תצוגה')}</span>
                <input
                  value={form.display_name}
                  onChange={(event) => setForm({ ...form, display_name: event.target.value })}
                />
              </label>
              <label className="auth-field">
                <span>{t('Role', 'תפקיד')}</span>
                <select value={form.role} onChange={(event) => setForm({ ...form, role: event.target.value })}>
                  <option value="viewer">{roleLabel('viewer', locale)}</option>
                  <option value="operator">{roleLabel('operator', locale)}</option>
                  <option value="admin">{roleLabel('admin', locale)}</option>
                </select>
              </label>
              <label className="auth-field">
                <span>{t('Affiliation', 'שיוך')}</span>
                <select value={form.affiliation} onChange={(event) => setForm({ ...form, affiliation: event.target.value })}>
                  <option value="company">{affiliationLabel('company', locale)}</option>
                  <option value="channel">{affiliationLabel('channel', locale)}</option>
                </select>
              </label>
              <label className="auth-field">
                <span>{t('Temporary password', 'סיסמה זמנית')}</span>
                <input
                  type="password"
                  dir="ltr"
                  autoComplete="new-password"
                  value={form.password}
                  onChange={(event) => setForm({ ...form, password: event.target.value })}
                />
              </label>
            </div>
            <p className="auth-hint">
              {t(
                'At least 10 characters; a password change is required at the first sign-in. The viewer role reads only, operator edits and runs, admin also manages accounts.',
                'לפחות 10 תווים; בכניסה הראשונה תידרש החלפת סיסמה. תפקיד צפייה מאפשר קריאה בלבד, תפעול מאפשר עריכה והרצה, וניהול מוסיף ניהול חשבונות.',
              )}
            </p>
            <p className="auth-hint">
              {t(
                'A channel-affiliated account cannot manage calendar events or event pricing.',
                'חשבון המשויך לערוץ אינו יכול לנהל אירועים ביומן או תמחור אירועים.',
              )}
            </p>
            {formError && (
              <p className="auth-error" role="alert">
                {formError}
              </p>
            )}
            <div className="auth-actions">
              <button
                type="submit"
                className="auth-primary"
                disabled={busy || form.username.trim() === '' || form.password === ''}
              >
                {t('Create account', 'יצירת חשבון')}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}

export default UserAdminDialog;
