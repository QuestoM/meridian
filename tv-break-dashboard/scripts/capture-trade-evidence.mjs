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
  { id: 'day-versions', path: '/?broadcast=versions#Broadcast' },
  { id: 'forecast-stage', path: '/?broadcast=forecast#Broadcast' },
  { id: 'today', path: '/#Today' },
];

// Deep states are addressed through the same query-string law: the panel reads
// `agreement=<id>` and derives the screen from the record's status (in_review
// with a document opens review; anything else opens the record). Ids are
// store-generated, so they are DISCOVERED from the live API rather than
// hardcoded: a reseeded store keeps the harness honest instead of breaking it.
async function discoverAgreementRoutes(apiBase) {
  try {
    const res = await fetch(`${apiBase}/api/trade/agreements`);
    if (!res.ok) return [];
    const body = await res.json();
    const rows = body.agreements || [];
    const routes = [];
    const reviewRow = rows.find((r) => r.status === 'in_review' && Number(r.documents) > 0);
    if (reviewRow) {
      routes.push({
        id: 'agreement-review',
        path: `/?clients=agreements&agreement=${reviewRow.agreement_id}#Commercial`,
      });
    }
    const approvedRow = rows.find((r) => r.status === 'approved');
    if (approvedRow) {
      routes.push({
        id: 'agreement-record-approved',
        path: `/?clients=agreements&agreement=${approvedRow.agreement_id}#Commercial`,
      });
    }
    const draftRow = rows.find((r) => r.status === 'draft');
    if (draftRow) {
      routes.push({
        id: 'agreement-record-draft',
        path: `/?clients=agreements&agreement=${draftRow.agreement_id}#Commercial`,
      });
    }
    routes.push({
      id: 'agreement-not-found',
      path: '/?clients=agreements&agreement=agr-000000dead#Commercial',
    });
    return routes;
  } catch {
    return [];
  }
}

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

  // The preview server proxies /api to the engine, so discovery rides the
  // same origin the screenshots use.
  const allRoutes = [...ROUTES, ...(await discoverAgreementRoutes(BASE))];
  process.stdout.write(`routes: ${allRoutes.map((r) => r.id).join(', ')}\n`);

  for (const size of SIZES) {
    for (const route of allRoutes) {
      // Filter BEFORE creating a target: a skipped route that already opened
      // one leaks a blank page per skip and never closes it.
      if (ONLY && !route.id.includes(ONLY)) continue;
      const target = await send(ws, 'Target.createTarget', { url: 'about:blank' });
      const { sessionId } = await send(ws, 'Target.attachToTarget', {
        targetId: target.targetId, flatten: true,
      });

      const problems = { console: [], failed: [] };
      // Every API round-trip, not only the failed ones: when a page reads
      // offline the question is always "which request, with what status, in
      // what order" — a failures-only log cannot answer it.
      const apiLog = [];
      const apiPending = {};
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
        if (data.method === 'Network.requestWillBeSent' && data.params.request.url.includes('/api/')) {
          apiPending[data.params.requestId] = data.params.request.url.replace(BASE, '');
        }
        if (data.method === 'Network.responseReceived' && apiPending[data.params.requestId]) {
          apiLog.push(`${data.params.response.status} ${apiPending[data.params.requestId]}`);
          delete apiPending[data.params.requestId];
        }
        if (data.method === 'Network.loadingFailed') {
          const named = apiPending[data.params.requestId];
          if (named) {
            apiLog.push(`FAIL(${data.params.errorText}) ${named}`);
            delete apiPending[data.params.requestId];
          }
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

      // Wait for the app's own liveness signal, not a fixed clock. A fixed
      // sleep photographed the loading state and the offline chip whenever a
      // parallel test run starved the API for a few seconds, and reported
      // success — the requests completed right after the shutter. The shell
      // states its own condition (the connection chip), so the harness reads
      // it; the deadline keeps a genuinely stuck page capturable, and the
      // report says which of the two the picture shows.
      const waitAlive = async (budgetMs) => {
        const deadline = Date.now() + budgetMs;
        while (Date.now() < deadline) {
          await new Promise((r) => setTimeout(r, 700));
          const probe = await send(ws, 'Runtime.evaluate', {
            returnByValue: true,
            expression: `(() => {
              const chip = (document.querySelector('.connection-state') || {}).textContent || '';
              const busy = document.body.innerText.includes('טוען סביבת Kairos');
              return { alive: chip.includes('API חי'), busy };
            })()`,
          }, sessionId);
          const { alive, busy } = probe.result.value || {};
          if (alive && !busy) return true;
        }
        return false;
      };
      let settled = await waitAlive(30000);
      await new Promise((r) => setTimeout(r, 1200));
      if (!settled) {
        process.stdout.write(`  ! ${route.id}/${size.id}: page never reported API-alive; capturing as-is\n`);
      }

      // Full page: grow the viewport to the document, capture, then measure the
      // things a picture cannot tell you by itself.
      //
      // The viewport MUST be grown to the real height before any capture:
      // captureBeyondViewport does not paint far past the layout viewport, so
      // clipping a region beyond it photographs the background and reports
      // success. Caught on the review screen (16,712px): the harness showed a
      // beige void where four thousand pixels of term cards actually render.
      // The cap exists only to bound the PNG; a page that exceeds it is
      // reported so the void is never mistaken for the page.
      const metrics = await send(ws, 'Page.getLayoutMetrics', {}, sessionId);
      const trueHeight = Math.ceil(metrics.cssContentSize.height);
      const full = Math.min(trueHeight, 24000);
      if (trueHeight > full) {
        process.stdout.write(`  ! ${route.id}/${size.id}: page ${trueHeight}px exceeds capture cap ${full}px\n`);
      }
      await send(ws, 'Emulation.setDeviceMetricsOverride', {
        width: size.width, height: full, deviceScaleFactor: 1, mobile: false,
      }, sessionId);
      // The grow re-issues layout, so give the page a beat and confirm the
      // shutter still sees a settled app rather than a mid-flight one.
      if (settled) settled = await waitAlive(30000);
      await new Promise((r) => setTimeout(r, 900));

      // NO captureBeyondViewport on any shot. That flag resizes the surface
      // internally for the duration of the capture; the momentary dip crossed
      // the desktop gate's threshold, the gate replaced the tree, and the
      // measurement that followed photographed a freshly remounted app that
      // honestly said it had no data yet. The viewport is already grown to
      // the full page, so plain screenshots and in-viewport clips see
      // everything without touching the surface.
      const shot = await send(ws, 'Page.captureScreenshot', {
        format: 'png',
      }, sessionId);
      const stem = `${route.id}-${size.id}`;
      writeFileSync(join(OUT, `${stem}-full.png`), Buffer.from(shot.data, 'base64'));

      // Section slices, so a long page is readable rather than a thumbnail.
      // Slices are spread over the WHOLE page — first slice at the top, last
      // at the bottom — because a tall page's defects live at its end as often
      // as its start, and consecutive-from-the-top slices never reach it.
      const slices = Math.min(4, Math.max(1, Math.ceil(full / size.height)));
      for (let index = 0; index < slices; index += 1) {
        const y = slices === 1
          ? 0
          : Math.round((full - size.height) * (index / (slices - 1)));
        if (y >= full) break;
        const clipped = await send(ws, 'Page.captureScreenshot', {
          format: 'png',
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
        settled,
        api_requests: apiLog.slice(0, 40),
        api_unanswered: Object.values(apiPending).slice(0, 12),
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
