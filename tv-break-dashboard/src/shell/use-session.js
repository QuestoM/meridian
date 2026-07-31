import { useEffect } from 'react';
import { API_BASE } from './api';
import { fetchMe } from './Login';

// The session probe and the credentialed-fetch guard, in the order the shell
// mounted them, so effect order is unchanged by the split.
export function useSessionEffects(setAuth) {
  useEffect(() => {
    let active = true;
    fetchMe().then((result) => {
      if (!active) return;
      if (result.ok && result.data && result.data.auth_disabled) {
        setAuth({ status: 'open', user: null });
      } else if (result.ok && result.data && result.data.username) {
        setAuth({ status: 'ready', user: result.data });
      } else if (result.status === 0) {
        // Server unreachable: render the app and let its offline states tell
        // the truth about connectivity; there is no session to pretend about.
        setAuth({ status: 'open', user: null });
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
        const url = typeof input === 'string'
          ? input
          : (input && typeof input.url === 'string' ? input.url : String(input || ''));
        if (
          response.status === 401 &&
          url.includes('/api/') &&
          !url.includes('/api/auth/')
        ) {
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
