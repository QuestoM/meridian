import React from 'react';
import { CacheProvider } from '@emotion/react';
import { CssBaseline, ThemeProvider } from '@mui/material';
import Login, { ChangePasswordDialog } from './Login';
import { DirectionRoot } from './bidi';

// The three pre-workspace screens. Returns null once a session is settled, so
// the shell renders the workspace exactly as the single file did.
export function renderAuthScreen({ auth, setAuth, muiCache, theme, handleLoggedIn }) {
  if (auth.status === 'checking') {
    return (
      <CacheProvider value={muiCache}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <DirectionRoot locale="he" className="login-screen" lang="he">
            <div className="login-loading">
              <div className="login-brand-mark" aria-hidden="true">
                <span />
                <span />
                <span />
              </div>
              <span>רק רגע...</span>
            </div>
          </DirectionRoot>
        </ThemeProvider>
      </CacheProvider>
    );
  }

  if (auth.status === 'login') {
    return (
      <CacheProvider value={muiCache}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <Login onLoggedIn={handleLoggedIn} />
        </ThemeProvider>
      </CacheProvider>
    );
  }

  if (auth.status === 'ready' && auth.user && auth.user.must_change_password) {
    return (
      <CacheProvider value={muiCache}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <DirectionRoot locale="he" className="login-screen" lang="he">
            <ChangePasswordDialog
              locale="he"
              forced
              onDone={(user) =>
                setAuth({ status: 'ready', user: { ...auth.user, ...user, must_change_password: false } })
              }
            />
          </DirectionRoot>
        </ThemeProvider>
      </CacheProvider>
    );
  }
  return null;
}
