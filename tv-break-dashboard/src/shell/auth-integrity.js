const AUTH_401_EXEMPT_PATHS = new Set([
  '/api/auth/session',
  '/api/auth/login',
  '/api/auth/logout',
]);

function requestPath(input) {
  const value = typeof input === 'string'
    ? input
    : (input && typeof input.url === 'string' ? input.url : String(input || ''));
  try {
    return new URL(value, 'http://kairos.local').pathname;
  } catch {
    return '';
  }
}

export function workspaceSessionReady(auth) {
  return auth?.status === 'open'
    || (auth?.status === 'ready' && Boolean(auth?.user?.username));
}

export function shouldExpireSession(input, status) {
  if (status !== 401) return false;
  const path = requestPath(input);
  return path.startsWith('/api/') && !AUTH_401_EXEMPT_PATHS.has(path);
}
