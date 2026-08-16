import React, { useMemo, useState } from 'react';
import TVBreakDashboard from './TVBreakDashboard.jsx';
import { createKairosTheme, rtlCache } from './theme';
import { renderAuthScreen } from './auth-screens';
import { useSessionEffects } from './use-session';
import DesktopGate, { useDesktopSupport } from './desktop-gate';

function SessionBoundary() {
  const [auth, setAuth] = useState({ status: 'checking', user: null });
  const theme = useMemo(() => createKairosTheme('rtl'), []);

  useSessionEffects(setAuth);

  const authScreen = renderAuthScreen({
    auth,
    setAuth,
    muiCache: rtlCache,
    theme,
    handleLoggedIn: (user) => setAuth({ status: 'ready', user }),
  });

  if (authScreen) return authScreen;
  return <TVBreakDashboard auth={auth} setAuth={setAuth} />;
}

function App() {
  const desktopSupported = useDesktopSupport();

  // The session and data trees deliberately live below this branch. A phone or
  // tablet gets one complete, useful gate rather than a hidden console that
  // still authenticates and starts protected requests behind the message.
  if (!desktopSupported) return <DesktopGate />;
  return <SessionBoundary />;
}

export default App;
