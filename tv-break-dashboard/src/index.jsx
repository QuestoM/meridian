import '@fontsource/ibm-plex-sans/latin-400.css';
import '@fontsource/ibm-plex-sans/latin-500.css';
import '@fontsource/ibm-plex-sans/latin-600.css';
import '@fontsource/ibm-plex-sans-hebrew/hebrew-400.css';
import '@fontsource/ibm-plex-sans-hebrew/hebrew-500.css';
import '@fontsource/ibm-plex-sans-hebrew/hebrew-600.css';
import '@fontsource/ibm-plex-mono/latin-400.css';
import '@fontsource/ibm-plex-mono/latin-500.css';
import notoHebrewVariableUrl from './assets/fonts/noto-sans-hebrew-hebrew-wght-normal.woff2?url';
import latinRegularUrl from '@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-400-normal.woff2?url';
import latinSemiboldUrl from '@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-600-normal.woff2?url';
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './shell/App.jsx';
import './assets/fonts/kairos-fonts.css';
import './tokens.css';
import './shell/styles.css';
import './shell/styles-planning-canvas.css';
import './shell/styles-timeline.css';
import './shell/styles-schedule-editor.css';
import './shell/styles-inventory.css';
import './shell/styles-inspection.css';
import './shell/styles-workspaces.css';
import './shell/styles-settings.css';
import './shell/styles-commercial.css';
import './shell/styles-print.css';
import './shell/styles-governance.css';
import './shell/styles-insights.css';
import './shell/styles-money.css';
// After the reachable shared sheets: the card owns the inset of everything
// inside it, and where it restates a legacy rule, the card is the one home.
import './shell/card.css';
import './studio/studio.css';
import './studio/studio-workspaces.css';
import './shell/studio-shell.css';
import './shell/shell-continuity.css';
import './shell/desktop-gate.css';
import './studio/typography.css';

[
  ['hebrew-variable', notoHebrewVariableUrl],
  ['latin-regular', latinRegularUrl],
  ['latin-semibold', latinSemiboldUrl],
].forEach(([id, href]) => {
  if (document.head.querySelector(`[data-kairos-font="${id}"]`)) return;
  const link = document.createElement('link');
  link.rel = 'preload';
  link.as = 'font';
  link.type = 'font/woff2';
  link.crossOrigin = 'anonymous';
  link.href = href;
  link.dataset.kairosFont = id;
  document.head.appendChild(link);
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
