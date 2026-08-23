// The dashboard's static server + API proxy, containerized.
//
// This mirrors the operator's own local topology exactly - a static server in
// front of uvicorn - because that is the shape every test and every demo has
// run against. The cloud gets the same shape, not a novel one.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';

const [dist, port, api] = process.argv.slice(2);
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png',
  '.woff2': 'font/woff2', '.ico': 'image/x-icon', '.map': 'application/json',
  '.webmanifest': 'application/manifest+json' };

createServer(async (req, res) => {
  if (req.url === '/healthz') { res.writeHead(200); return res.end('ok'); }
  if (req.url.startsWith('/api/') || req.url.startsWith('/auth/')) {
    const body = ['GET', 'HEAD'].includes(req.method) ? undefined
      : await new Promise((r) => { const c = []; req.on('data', (d) => c.push(d)); req.on('end', () => r(Buffer.concat(c))); });
    try {
      const upstream = await fetch(api + req.url, {
        method: req.method,
        headers: { ...req.headers, host: new URL(api).host },
        body,
        redirect: 'manual',
      });
      const headers = {};
      upstream.headers.forEach((v, k) => { if (k !== 'transfer-encoding' && k !== 'content-encoding') headers[k] = v; });
      // Set-Cookie needs the raw multi-value form or the auth session breaks.
      const cookies = upstream.headers.getSetCookie?.() || [];
      if (cookies.length) headers['set-cookie'] = cookies;
      res.writeHead(upstream.status, headers);
      res.end(Buffer.from(await upstream.arrayBuffer()));
    } catch (error) { res.writeHead(502); res.end(String(error)); }
    return;
  }
  const path = normalize(req.url.split('?')[0]).replace(/^(\.\.[/\\])+/, '');
  for (const candidate of [join(dist, path), join(dist, 'index.html')]) {
    try {
      const data = await readFile(candidate);
      res.writeHead(200, { 'content-type': TYPES[extname(candidate)] || 'application/octet-stream' });
      return res.end(data);
    } catch { /* fall through to index.html, then 404 */ }
  }
  res.writeHead(404); res.end('not found');
}).listen(Number(port), '0.0.0.0', () => console.log('kairos dashboard on', port, '->', api));
