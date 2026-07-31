import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './shell/App.jsx';
import './tokens.css';
import './shell/styles.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
