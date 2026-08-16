import React from 'react';
import { Menu, MenuItem, Tooltip } from '@mui/material';
import { Button } from '../studio/actions';
import { ChevronDown, Info, KeyRound, LogOut, Users } from 'lucide-react';
import { pageText } from './format';
import { DOMAIN_DEFINITIONS } from './nav';
import { MabatIcon, KairosMark } from './kairos-icons';
import { roleLabel } from './Login';
import { Pressable } from './dom-controls';

function operatorInitials(name) {
  const parts = String(name || '').trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return '?';
  const first = parts[0][0] || '';
  const second = parts.length > 1 ? parts[parts.length - 1][0] || '' : '';
  return (first + second).toUpperCase() || '?';
}

export function renderSideRail({
  copy,
  locale,
  activeDomain,
  setActiveView,
  assistantOpen,
  auth,
  canAccessModel,
  userMenuAnchor,
  setUserMenuAnchor,
  setPasswordDialogOpen,
  setAccountsDialogOpen,
  handleLogout,
}) {
  return (
      <aside className="side-rail" aria-label={pageText(locale, 'Kairos primary navigation', 'הניווט הראשי של Kairos')}>
        <div className="brand-lockup">
          <KairosMark size={34} />
          <strong className="brand-wordmark">KAIROS</strong>
          <small className="brand-workspace">{copy.workspace}</small>
        </div>

        <nav className="primary-nav" aria-label={pageText(locale, 'Work domains', 'תחומי עבודה')}>
          {DOMAIN_DEFINITIONS.map(({ id, icon: Icon, en, he }) => {
            const isActive = id === activeDomain;
            return (
            <Button
              key={id}
              variant="text"
              className={isActive ? 'nav-item active' : 'nav-item'}
              aria-current={isActive ? 'page' : undefined}
              onClick={() => setActiveView(id)}
            >
              <span className="nav-icon" aria-hidden="true"><Icon size={19} /></span>
              <span className="nav-text">{locale === 'he' ? he : en}</span>
            </Button>
            );
          })}
        </nav>

        <div className="rail-tools">
          <Button
            variant="text"
            className={assistantOpen ? 'nav-item assistant-entry active' : 'nav-item assistant-entry'}
            aria-pressed={assistantOpen}
            onClick={() => setActiveView('Assistant')}
          >
            <span className="nav-icon" aria-hidden="true"><MabatIcon size={20} /></span>
            <span className="nav-text">{pageText(locale, 'Mabat assistant', 'מבט, העוזר')}</span>
          </Button>
        </div>

        {auth.status === 'ready' && auth.user ? (
          <>
            <Pressable
              type="button"
              className="operator-card"
              onClick={(event) => setUserMenuAnchor(event.currentTarget)}
              aria-haspopup="menu"
            >
              <span className="operator-avatar">{operatorInitials(auth.user.display_name || auth.user.username)}</span>
              <div>
                <strong>{auth.user.display_name || auth.user.username}</strong>
                <small>{roleLabel(auth.user.role, locale)}</small>
              </div>
              <ChevronDown size={14} />
            </Pressable>
            <Menu anchorEl={userMenuAnchor} open={Boolean(userMenuAnchor)} onClose={() => setUserMenuAnchor(null)}>
              <MenuItem
                onClick={() => {
                  setUserMenuAnchor(null);
                  setPasswordDialogOpen(true);
                }}
              >
                <KeyRound size={16} className="menu-item-icon" />
                {pageText(locale, 'Change password', 'החלפת סיסמה')}
              </MenuItem>
              {auth.user.role === 'admin' && (
                <MenuItem
                  onClick={() => {
                    setUserMenuAnchor(null);
                    setAccountsDialogOpen(true);
                  }}
                >
                  <Users size={16} className="menu-item-icon" />
                  {pageText(locale, 'Manage accounts', 'ניהול חשבונות')}
                </MenuItem>
              )}
              {canAccessModel && (
                <MenuItem
                  onClick={() => {
                    setUserMenuAnchor(null);
                    setActiveView('Model');
                  }}
                >
                  <MabatIcon size={16} className="menu-item-icon" />
                  {pageText(locale, 'Company model', 'מודל החברה')}
                </MenuItem>
              )}
              <MenuItem onClick={handleLogout}>
                <LogOut size={16} className="menu-item-icon" />
                {pageText(locale, 'Sign out', 'יציאה מהמערכת')}
              </MenuItem>
            </Menu>
          </>
        ) : (
          <Tooltip
            title={pageText(
              locale,
              'To set up sign-in and roles, run python scripts/init_auth.py on the server.',
              'להגדרת כניסה ותפקידים הריצו בשרת את python scripts/init_auth.py.',
            )}
            arrow
            placement="bottom"
          >
            <div className="operator-card operator-open">
              <span className="operator-avatar">?</span>
              <div>
                <strong>{pageText(locale, 'Open access', 'גישה פתוחה')}</strong>
                <small>{pageText(locale, 'Sign-in is not set up yet', 'כניסה למערכת טרם הוגדרה')}</small>
              </div>
              <Info size={14} />
            </div>
          </Tooltip>
        )}
      </aside>
  );
}
