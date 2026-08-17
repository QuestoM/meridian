// Look at the trade surface, at real sizes, in every state that matters.
//
// A first-viewport screenshot hides most of a page, so every route here is
// captured FULL PAGE (the viewport is grown to the document height) and then in
// section slices, at two desktop widths. The point is not to prove the page
// exists: it is to see the things only a picture shows — a sentence wrapping
// badly, a figure colliding with its label, a Latin string leaking into a
// Hebrew line, a card whose inset collapsed, a control under the target size.
//
// Console errors and failed requests are collected per route and printed, since
// a screen that looks right while its network fails is not right.
//
//   node scripts/capture-trade-evidence.mjs --base-url http://127.0.0.1:3001 \
//        --out /tmp/trade-evidence
//
// Chrome is driven over the DevTools protocol directly: no Playwright, nothing
// installed, and the same binary a reviewer has open.

import { mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { join } from 'node:path';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

function arg(name, fallback) {
  const hit = process.argv.find((item) => item.startsWith(`--${name}=`));
  if (hit) return hit.split('=').slice(1).join('=');
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

const BASE = arg('base-url', 'http://127.0.0.1:3001').replace(/\/$/, '');
const OUT = arg('out', '/tmp/trade-evidence');
const ONLY = arg('only', '');

const SIZES = [
  { id: 'wide', width: 1728, height: 900 },
  { id: 'laptop', width: 1280, height: 720 },
];

// Every route is an address a person can actually reach, so a capture that
// cannot load one is a real navigation defect rather than a harness gap.
//
// THE ADDRESS FORM IS NOT A DETAIL. The shell puts the DOMAIN in the hash and
// every scoped parameter in the REAL query string (nav.js navigationUrl), so
// `#Commercial?clients=agreements` silently lands on Today: the hash is read as
// the literal domain name and the parameter is never seen. Caught by looking at
// the first screenshot, which showed the Today page under an agreements
// filename — a harness that gets this wrong photographs the wrong screen and
// reports success.
const ROUTES = [
  { id: 'agreements-list', path: '/?clients=agreements#Commercial' },
  { id: 'commercial-clients', path: '/?clients=clients#Commercial' },
  { id: 'plan-board', path: '/?plan=board#Plan' },
  { id: 'plan-compare', path: '/?plan=compare#Plan' },
  { id: 'today', path: '/#Today' },
];

let messageId = 0;
function send(ws, method, params = {}, sessionId) {
  const id = ++messageId;
  ws.send(JSON.stringify({ id, method, params, sessionId }));
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`timeout on ${method}`)), 45000);
    const onMessage = (raw) => {
      const data = JSON.parse(raw.toString());
      if (data.id !== id) return;
      clearTimeout(timer);
      ws.off('message', onMessage);
      if (data.error) reject(new Error(`${method}: ${data.error.message}`));
      else resolve(data.result);
    };
    ws.on('message', onMessage);
  });
}

async function main() {
  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(OUT, { recursive: true });

  const profile = join(OUT, '.chrome-profile');
  const chrome = spawn(CHROME, [
    '--headless=new',
    '--remote-debugging-port=0',
    `--user-data-dir=${profile}`,
    '--no-first-run',
    '--disable-gpu',
    '--hide-scrollbars',
    '--force-device-scale-factor=1',
  ], { stdio: ['ignore', 'ignore', 'pipe'] });

  const wsUrl = await new Promise((resolve, reject) => {
    let buffer = '';
    const timer = setTimeout(() => reject(new Error('chrome did not report a debug url')), 20000);
    chrome.stderr.on('data', (chunk) => {
      buffer += chunk.toString();
      const match = buffer.match(/ws:\/\/[^\s]+/);
      if (match) { clearTimeout(timer); resolve(match[0]); }
    });
  });

  // Node's own WebSocket, as the sibling harness uses: no dependency to install
  // and no version to keep in step. Its API is the browser one, so the handlers
  // below are addEventListener rather than the node-style .on().
  const ws = new WebSocket(wsUrl);
  const listeners = new Set();
  ws.addEventListener('message', (event) => {
    listeners.forEach((fn) => fn(event.data));
  });
  ws.on = (_event, fn) => listeners.add(fn);
  ws.off = (_event, fn) => listeners.delete(fn);
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve, { once: true });
    ws.addEventListener('error', () => reject(new Error('could not attach to Chrome')), { once: true });
  });

  const report = { base: BASE, captured_at: new Date().toISOString(), routes: [] };

  for (const size of SIZES) {
    for (const route of ROUTES) {
      if (ONLY && !route.id.includes(ONLY)) continue;
      const target = await send(ws, 'Target.createTarget', { url: 'about:blank' });
      const { sessionId } = await send(ws, 'Target.attachToTarget', {
        targetId: target.targetId, flatten: true,
      });

      const problems = { console: [], failed: [] };
      const onEvent = (raw) => {
        const data = JSON.parse(raw.toString());
        if (data.sessionId !== sessionId) return;
        if (data.method === 'Runtime.consoleAPICalled' && data.params.type === 'error') {
          problems.console.push((data.params.args || [])
            .map((a) => a.value ?? a.description ?? a.type).join(' ').slice(0, 300));
        }
        if (data.method === 'Runtime.exceptionThrown') {
          problems.console.push(String(
            data.params.exceptionDetails?.exception?.description
            ?? data.params.exceptionDetails?.text,
          ).slice(0, 300));
        }
        if (data.method === 'Network.loadingFailed') {
          problems.failed.push(`${data.params.type}: ${data.params.errorText}`);
        }
        if (data.method === 'Network.responseReceived' && data.params.response.status >= 400) {
          problems.failed.push(`${data.params.response.status} ${data.params.response.url}`);
        }
      };
      ws.on('message', onEvent);

      await send(ws, 'Runtime.enable', {}, sessionId);
      await send(ws, 'Network.enable', {}, sessionId);
      await send(ws, 'Page.enable', {}, sessionId);
      await send(ws, 'Emulation.setDeviceMetricsOverride', {
        width: size.width, height: size.height, deviceScaleFactor: 1, mobile: false,
      }, sessionId);

      await send(ws, 'Page.navigate', { url: `${BASE}${route.path}` }, sessionId);
      await new Promise((r) => setTimeout(r, 9000));

      // Full page: grow the viewport to the document, capture, then measure the
      // things a picture cannot tell you by itself.
      const metrics = await send(ws, 'Page.getLayoutMetrics', {}, sessionId);
      const full = Math.min(Math.ceil(metrics.cssContentSize.height), 12000);
      await send(ws, 'Emulation.setDeviceMetricsOverride', {
        width: size.width, height: full, deviceScaleFactor: 1, mobile: false,
      }, sessionId);
      await new Promise((r) => setTimeout(r, 900));

      const shot = await send(ws, 'Page.captureScreenshot', {
        format: 'png', captureBeyondViewport: true,
      }, sessionId);
      const stem = `${route.id}-${size.id}`;
      writeFileSync(join(OUT, `${stem}-full.png`), Buffer.from(shot.data, 'base64'));

      // Section slices, so a long page is readable rather than a thumbnail.
      const slices = Math.min(4, Math.max(1, Math.ceil(full / size.height)));
      for (let index = 0; index < slices; index += 1) {
        const y = index * size.height;
        if (y >= full) break;
        const clipped = await send(ws, 'Page.captureScreenshot', {
          format: 'png',
          captureBeyondViewport: true,
          clip: { x: 0, y, width: size.width, height: Math.min(size.height, full - y), scale: 1 },
        }, sessionId);
        writeFileSync(join(OUT, `${stem}-section-${String(index + 1).padStart(2, '0')}.png`),
          Buffer.from(clipped.data, 'base64'));
      }

      const measured = await send(ws, 'Runtime.evaluate', {
        returnByValue: true,
        expression: `(() => {
          const out = { hardBreaks: 0, nbsp: 0, small: [], overflowX: 0, dirAttrs: 0, latinInHebrew: [] };
          const scope = document.querySelector('main') || document.body;
          out.hardBreaks = scope.querySelectorAll('br').length;
          out.nbsp = (scope.textContent.match(/\\u00a0/g) || []).length;
          out.dirAttrs = scope.querySelectorAll('[dir]').length;
          out.overflowX = document.documentElement.scrollWidth > window.innerWidth + 2 ? 1 : 0;
          scope.querySelectorAll('button, a[href], [role="tab"], input, select').forEach((el) => {
            const r = el.getBoundingClientRect();
            if (r.width > 0 && r.height > 0 && r.height < 44) {
              out.small.push((el.textContent || el.tagName).trim().slice(0, 40) + ' @' + Math.round(r.height));
            }
          });
          out.title = (document.querySelector('main h1, main h2') || {}).textContent || '';
          out.text = scope.innerText.slice(0, 400);
          return out;
        })()`,
      }, sessionId);

      report.routes.push({
        route: route.id, size: size.id, url: `${BASE}${route.path}`,
        full_height: full, sections: slices,
        console_errors: problems.console.slice(0, 8),
        failed_requests: problems.failed.slice(0, 8),
        measured: measured.result.value,
      });
      ws.off('message', onEvent);
      await send(ws, 'Target.closeTarget', { targetId: target.targetId });
      process.stdout.write(`captured ${stem} (${full}px, ${slices} sections)\n`);
    }
  }

  writeFileSync(join(OUT, 'report.json'), JSON.stringify(report, null, 1));
  ws.close();
  chrome.kill();

  const errors = report.routes.filter((r) => r.console_errors.length);
  const failures = report.routes.filter((r) => r.failed_requests.length);
  const breaks = report.routes.filter((r) => r.measured?.hardBreaks || r.measured?.nbsp);
  const smalls = report.routes.filter((r) => (r.measured?.small || []).length);
  const overflow = report.routes.filter((r) => r.measured?.overflowX);
  console.log(`\nwrote ${OUT}/report.json`);
  console.log(`routes with console errors:   ${errors.length}`);
  console.log(`routes with failed requests:  ${failures.length}`);
  console.log(`routes with <br>/&nbsp;:      ${breaks.length}`);
  console.log(`routes with sub-44px targets: ${smalls.length}`);
  console.log(`routes scrolling sideways:    ${overflow.length}`);
  for (const route of errors) console.log(`  ! ${route.route}/${route.size}: ${route.console_errors[0]}`);
  for (const route of failures) console.log(`  ! ${route.route}/${route.size}: ${route.failed_requests[0]}`);
  for (const route of breaks) console.log(`  ! ${route.route}/${route.size}: ${route.measured.hardBreaks} <br>, ${route.measured.nbsp} nbsp`);
}

main().catch((error) => { console.error(error.message); process.exit(1); });
