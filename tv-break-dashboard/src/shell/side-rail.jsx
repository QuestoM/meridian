import React from 'react';
import { List, ListItemButton, ListItemIcon, ListItemText, Menu, MenuItem, Tooltip } from '@mui/material';
import { ChevronDown, Info, KeyRound, LogOut, Users } from 'lucide-react';
import { pageText } from './format';
import { navItems } from './nav';
import { roleLabel } from './Login';

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
  activeView,
  setActiveView,
  assistantOpen,
  auth,
  userMenuAnchor,
  setUserMenuAnchor,
  setPasswordDialogOpen,
  setAccountsDialogOpen,
  handleLogout,
}) {
  return (
      <aside className="side-rail" aria-label="Kairos navigation">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <strong>Kairos</strong>
            <small>{copy.workspace}</small>
          </div>
        </div>

        <List component="nav" className="primary-nav" disablePadding>
          {navItems.map(([label, Icon]) => {
            // The Assistant entry reflects the dock: lit while the dock is
            // open on any page, and clicking it opens or focuses the dock.
            const isActive = label === 'Assistant' ? assistantOpen : label === activeView;
            return (
            <ListItemButton
              key={label}
              component="button"
              className={isActive ? 'nav-item active' : 'nav-item'}
              type="button"
              selected={isActive}
              disableRipple
              aria-current={label === activeView ? 'page' : undefined}
              onClick={() => setActiveView(label)}
            >
              <ListItemIcon className="nav-icon">
                <Icon size={16} strokeWidth={1.8} />
              </ListItemIcon>
              <ListItemText className="nav-text" disableTypography primary={<span>{copy.nav[label]}</span>} />
            </ListItemButton>
            );
          })}
        </List>

        {auth.status === 'ready' && auth.user ? (
          <>
            <button
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
            </button>
            <Menu anchorEl={userMenuAnchor} open={Boolean(userMenuAnchor)} onClose={() => setUserMenuAnchor(null)}>
              <MenuItem
                onClick={() => {
                  setUserMenuAnchor(null);
                  setPasswordDialogOpen(true);
                }}
              >
                <KeyRound size={14} style={{ marginInlineEnd: 8 }} />
                {pageText(locale, 'Change password', 'החלפת סיסמה')}
              </MenuItem>
              {auth.user.role === 'admin' && (
                <MenuItem
                  onClick={() => {
                    setUserMenuAnchor(null);
                    setAccountsDialogOpen(true);
                  }}
                >
                  <Users size={14} style={{ marginInlineEnd: 8 }} />
                  {pageText(locale, 'Manage accounts', 'ניהול חשבונות')}
                </MenuItem>
              )}
              <MenuItem onClick={handleLogout}>
                <LogOut size={14} style={{ marginInlineEnd: 8 }} />
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
