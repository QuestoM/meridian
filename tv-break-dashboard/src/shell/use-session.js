import { useEffect } from 'react';
import { API_BASE } from './api';
import { fetchMe, setSessionProbeIssue } from './Login';
import { shouldExpireSession } from './auth-integrity';

// The session probe and the credentialed-fetch guard, in the order the shell
// mounted them, so effect order is unchanged by the split.
export function useSessionEffects(setAuth) {
  useEffect(() => {
    let active = true;
    fetchMe().then((result) => {
      if (!active) return;
      setSessionProbeIssue(result.status === 0 ? 'offline' : result.status === 503 ? 'setup' : '');
      if (result.ok && result.data && result.data.auth_disabled === true) {
        setAuth({ status: 'open', user: null });
      } else if (result.ok && result.data && result.data.authenticated === true && result.data.username) {
        setAuth({ status: 'ready', user: result.data });
      } else if (result.status === 0 || result.status === 503) {
        // Authority could not be verified. Stay outside the workspace and let
        // the dedicated retry screen explain the failure without disclosing it.
        setAuth({ status: 'login', user: null });
      } else {
        setAuth({ status: 'login', user: null });
      }
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const originalFetch = window.fetch;
    window.fetch = async (input, init) => {
      let response;
      try {
        const url = typeof input === 'string'
          ? input
          : (input && typeof input.url === 'string' ? input.url : String(input || ''));
        // When API_BASE is '' (same-origin proxy), startsWith('') is always true;
        // only treat real /api/ paths (or an absolute override host) as API traffic.
        const targetsApi = url.includes('/api/') || (API_BASE !== '' && url.startsWith(API_BASE));
        if (targetsApi && !(input instanceof Request)) {
          response = await originalFetch(input, { ...init, credentials: (init && init.credentials) || 'include' });
        } else if (targetsApi && input instanceof Request && input.credentials === 'same-origin') {
          response = await originalFetch(new Request(input, { credentials: 'include' }), init);
        } else {
          response = await originalFetch(input, init);
        }
      } catch (err) {
        throw err;
      }
      try {
        if (shouldExpireSession(input, response.status)) {
          setAuth((current) => (current.status === 'login' ? current : { status: 'login', user: null }));
        }
      } catch {
        // The guard must never break a fetch.
      }
      return response;
    };
    return () => {
      window.fetch = originalFetch;
    };
  }, []);
}
