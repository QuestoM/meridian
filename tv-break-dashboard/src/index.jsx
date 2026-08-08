import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './shell/App.jsx';
import './tokens.css';
import './shell/styles.css';
// After styles.css: the card owns the inset of everything inside it, and where
// it restates something the shell sheet also says, the card is the one home.
import './shell/card.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
