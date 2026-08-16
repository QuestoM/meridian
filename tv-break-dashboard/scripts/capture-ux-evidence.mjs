#!/usr/bin/env node
// Dependency-free Kairos UX evidence recorder. It launches a private headless
// Chrome on an ephemeral CDP port, performs GET-only navigation, and never
// clicks product controls. See capture-ux-evidence.md for usage and output.
import { spawn } from 'node:child_process';
import {
  existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { aggregateMarkdown, buildAggregate } from './ux-evidence-report.mjs';
import { MOTION_AUDIT, PAGE_AUDIT } from './ux-evidence-audits.mjs';

const ROUTES = [
  ['today', '/#Today'],
  ['plan-objective', '/?plan=objective#Plan'],
  ['plan-run', '/?plan=run#Plan'],
  ['plan-compare', '/?plan=compare#Plan'],
  ['plan-publish', '/?plan=publish#Plan'],
  ['plan-supply', '/?plan=supply#Plan'],
  ['plan-board', '/?plan=board#Plan'],
  ['broadcast-day', '/?broadcast=day#Broadcast'],
  ['broadcast-pods', '/?broadcast=pods#Broadcast'],
  ['broadcast-library', '/?broadcast=library#Broadcast'],
  ['broadcast-decisions', '/?broadcast=decisions#Broadcast'],
  ['commercial-clients', '/?clients=clients#Commercial'],
  ['commercial-money', '/?clients=money#Commercial'],
  ['commercial-campaigns', '/?clients=campaigns#Commercial'],
  ['commercial-pacing', '/?clients=pacing#Commercial'],
  ['commercial-advertisers', '/?clients=advertisers#Commercial'],
  ['commercial-agencies', '/?clients=agencies#Commercial'],
  ['sources-inputs', '/?sources=inputs#Sources'],
  ['sources-files', '/?sources=files#Sources'],
  ['sources-reports', '/?sources=downloads#Sources'],
  ['governance-restrictions', '/?rules=restrictions#Governance'],
  ['governance-licence', '/?rules=licence#Governance'],
  ['governance-rate-card', '/?rules=rate_card#Governance'],
  ['governance-calendar', '/?rules=calendar#Governance'],
  ['governance-channel', '/?rules=channel#Governance'],
  ['governance-levers', '/?rules=levers#Governance'],
  ['history', '/#History'],
  ['model-gates', '/?modelSection=gates#Model'],
  ['model-coverage', '/?modelSection=coverage#Model'],
  ['model-drift', '/?modelSection=drift#Model'],
  ['model-candidates', '/?modelSection=candidates#Model'],
  ['model-training', '/?modelSection=training#Model'],
  ['model-versions', '/?modelSection=versions#Model'],
  ['model-provenance', '/?modelSection=provenance#Model'],
].map(([slug, address]) => ({ slug, address }));
if (ROUTES.length !== 34 || new Set(ROUTES.map(({ slug }) => slug)).size !== 34) throw new Error('canonical route registry must contain 34 unique slugs');
const VIEWPORTS = [{ slug: '1280x720', width: 1280, height: 720 }, { slug: '1728x900', width: 1728, height: 900 }];
const LOCALES = [{ locale: 'he', direction: 'rtl' }, { locale: 'en', direction: 'ltr' }];
const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome', '/usr/bin/chromium',
].filter(Boolean);
const sleep = (ms) => new Promise((accept) => setTimeout(accept, ms));
const ACTIVE_BROWSERS = new Set();

function usage(exitCode = 0) {
  const slugs = ROUTES.map(({ slug }) => slug).join(', ');
  process.stdout.write(`Usage: node scripts/capture-ux-evidence.mjs --base-url URL --out DIR [options]\n\n` +
    `  --routes a,b       capture only named canonical slugs\n` +
    `  --all              capture all canonical routes (use with --gate for the full suite)\n` +
    `  --gate             capture the 1024x768 desktop gate; alone, captures only the gate\n` +
    `  --resume           replace selected captures, then aggregate every existing canonical report\n` +
    `  --timeout-ms N     per-page settle ceiling (default 30000)\n` +
    `  --list-routes      print stable route slugs\n\nRoutes: ${slugs}\n`);
  process.exit(exitCode);
}

function parseArgs(argv) {
  const options = { timeoutMs: 30000, routeSlugs: [], routesExplicit: false, all: false, gate: false, resume: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const value = () => {
      const next = argv[index + 1];
      if (!next || next.startsWith('--')) throw new Error(`${arg} requires a value`);
      index += 1;
      return next;
    };
    if (arg === '--base-url') options.baseUrl = value();
    else if (arg === '--out') options.out = value();
    else if (arg === '--routes' || arg === '--route') {
      options.routesExplicit = true;
      options.routeSlugs.push(...value().split(',').map((item) => item.trim()).filter(Boolean));
    } else if (arg === '--timeout-ms') options.timeoutMs = Number(value());
    else if (arg === '--all') options.all = true;
    else if (arg === '--gate') options.gate = true;
    else if (arg === '--resume') options.resume = true;
    else if (arg === '--list-routes') { ROUTES.forEach(({ slug }) => process.stdout.write(`${slug}\n`)); process.exit(0); }
    else if (arg === '--help' || arg === '-h') usage();
    else throw new Error(`unknown option: ${arg}`);
  }
  if (!options.baseUrl || !options.out) usage(2);
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 1000) throw new Error('--timeout-ms must be at least 1000');
  options.baseUrl = new URL(options.baseUrl).href;
  options.out = resolve(options.out);
  const requested = new Set(options.routeSlugs);
  if (options.all && options.routesExplicit) throw new Error('--all and --routes are mutually exclusive');
  const unknown = [...requested].filter((slug) => !ROUTES.some((route) => route.slug === slug));
  if (unknown.length) throw new Error(`unknown route slug(s): ${unknown.join(', ')}`);
  options.routes = options.all ? ROUTES : options.routesExplicit ? ROUTES.filter(({ slug }) => requested.has(slug)) : (options.gate || options.resume ? [] : ROUTES);
  return options;
}

class Protocol {
  constructor(socket) {
    this.socket = socket;
    this.counter = 0;
    this.pending = new Map();
    this.listeners = new Set();
    socket.addEventListener('message', ({ data }) => this.receive(JSON.parse(data)));
  }
  receive(message) {
    if (message.id !== undefined && this.pending.has(message.id)) {
      const pair = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (message.error) pair.reject(new Error(`${pair.method}: ${JSON.stringify(message.error)}`));
      else pair.resolve(message.result);
      return;
    }
    this.listeners.forEach((listener) => listener(message));
  }
  send(method, params = {}, sessionId) {
    const id = ++this.counter;
    return new Promise((resolvePromise, reject) => {
      this.pending.set(id, { resolve: resolvePromise, reject, method });
      this.socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  }
  on(listener) { this.listeners.add(listener); return () => this.listeners.delete(listener); }
}

class PrivateChrome {
  constructor(windowSize) {
    this.windowSize = windowSize;
    this.profile = mkdtempSync(join(tmpdir(), 'kairos-ux-evidence-'));
    ACTIVE_BROWSERS.add(this);
  }
  async start() {
    const binary = CHROME_CANDIDATES.find((candidate) => existsSync(candidate));
    if (!binary) throw new Error('Chrome/Chromium not found; set CHROME_PATH');
    this.child = spawn(binary, [
      '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${this.profile}`,
      '--no-first-run', '--no-default-browser-check', '--disable-gpu', '--hide-scrollbars',
      '--force-device-scale-factor=1', `--window-size=${this.windowSize.width},${this.windowSize.height}`,
      '--window-position=0,0', 'about:blank',
    ], { stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    this.child.stderr.on('data', (chunk) => { stderr += String(chunk); });
    const marker = join(this.profile, 'DevToolsActivePort');
    for (let attempt = 0; attempt < 200; attempt += 1) {
      if (existsSync(marker)) {
        const [port, path] = readFileSync(marker, 'utf8').trim().split('\n');
        const socket = new WebSocket(`ws://127.0.0.1:${port}${path}`);
        await new Promise((accept, reject) => { socket.addEventListener('open', accept, { once: true }); socket.addEventListener('error', reject, { once: true }); });
        this.socket = socket;
        this.cdp = new Protocol(socket);
        return this;
      }
      if (this.child.exitCode !== null) throw new Error(`Chrome exited before CDP was ready: ${stderr.trim()}`);
      await sleep(100);
    }
    throw new Error(`Chrome never published a debugging port: ${stderr.trim()}`);
  }
  async stop() {
    try { this.socket?.close(); } catch { /* already closed */ }
    if (this.child && this.child.exitCode === null) {
      this.child.kill('SIGTERM');
      await Promise.race([new Promise((accept) => this.child.once('exit', accept)), sleep(3000)]);
      if (this.child.exitCode === null) this.child.kill('SIGKILL');
    }
    rmSync(this.profile, { recursive: true, force: true });
    ACTIVE_BROWSERS.delete(this);
  }
}

function localeBootstrap({ locale, direction }) {
  return `(() => {
    const wantedLocale = ${JSON.stringify(locale)};
    const wantedDirection = ${JSON.stringify(direction)};
    try { localStorage.setItem('kairos.locale', wantedLocale); } catch {}
    const originalFetch = window.fetch.bind(window);
    const endpoints = new Set(['/api/overview', '/api/settings']);
    const rewrite = (value) => {
      if (Array.isArray(value)) return value.map(rewrite);
      if (!value || typeof value !== 'object') return value;
      const next = {};
      for (const [key, nested] of Object.entries(value)) {
        next[key] = key === 'locale' ? wantedLocale : key === 'direction' ? wantedDirection : rewrite(nested);
      }
      return next;
    };
    const replaceResponse = async (response) => {
      if (!String(response.headers.get('content-type') || '').toLowerCase().includes('json')) return response;
      let payload;
      try { payload = rewrite(await response.clone().json()); } catch { return response; }
      const replacement = new Response(JSON.stringify(payload), {
        status: response.status, statusText: response.statusText, headers: new Headers(response.headers),
      });
      return new Proxy(replacement, { get(target, property) {
        if (property === 'url' || property === 'redirected' || property === 'type') return response[property];
        const found = Reflect.get(target, property, target);
        return typeof found === 'function' ? found.bind(target) : found;
      }});
    };
    window.fetch = async (input, init) => {
      const response = await originalFetch(input, init);
      const raw = typeof input === 'string' || input instanceof URL ? String(input) : input?.url;
      let path = '';
      try { path = new URL(raw, location.href).pathname; } catch {}
      return endpoints.has(path) ? replaceResponse(response) : response;
    };
  })();`;
}

function consoleValue(argument) {
  if (argument.value !== undefined) return typeof argument.value === 'string' ? argument.value : JSON.stringify(argument.value);
  return argument.unserializableValue || argument.description || argument.type;
}

function observe(cdp, sessionId) {
  const state = { requests: new Map(), pending: new Set(), httpErrors: [], failures: [], cancellations: [], consoleErrors: [], consoleWarnings: [], lastActivity: Date.now() };
  const pushConsole = (bucket, record) => { if (bucket.length < 200) bucket.push(record); };
  const release = (id) => { state.pending.delete(id); state.lastActivity = Date.now(); };
  const off = cdp.on((message) => {
    if (message.sessionId !== sessionId) return;
    const p = message.params || {};
    if (message.method === 'Network.requestWillBeSent') {
      const request = { url: p.request.url, method: p.request.method, type: p.type || 'Other', timestamp: p.timestamp };
      state.requests.set(p.requestId, request);
      if (/^https?:/.test(request.url)) state.pending.add(p.requestId);
      state.lastActivity = Date.now();
    } else if (message.method === 'Network.responseReceived') {
      const request = state.requests.get(p.requestId) || {};
      request.status = p.response.status; request.type = p.type; request.mimeType = p.response.mimeType;
      state.requests.set(p.requestId, request);
      if (p.response.status >= 400) state.httpErrors.push({ url: p.response.url, status: p.response.status, statusText: p.response.statusText, type: p.type });
      if (p.type === 'EventSource') release(p.requestId);
    } else if (message.method === 'Network.loadingFinished') release(p.requestId);
    else if (message.method === 'Network.loadingFailed') {
      const record = { ...(state.requests.get(p.requestId) || {}), errorText: p.errorText, blockedReason: p.blockedReason || null, canceled: Boolean(p.canceled) };
      (p.canceled || p.errorText === 'net::ERR_ABORTED' ? state.cancellations : state.failures).push(record);
      release(p.requestId);
    } else if (message.method === 'Network.webSocketCreated') release(p.requestId);
    else if (message.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(p.type)) {
      const record = { source: 'console', type: p.type, text: p.args.map(consoleValue).join(' '), timestamp: p.timestamp };
      pushConsole(p.type === 'error' ? state.consoleErrors : state.consoleWarnings, record);
    } else if (message.method === 'Runtime.exceptionThrown') {
      pushConsole(state.consoleErrors, { source: 'exception', text: p.exceptionDetails?.exception?.description || p.exceptionDetails?.text || 'Uncaught exception', timestamp: p.timestamp });
    } else if (message.method === 'Log.entryAdded' && ['error', 'warning'].includes(p.entry.level)) {
      pushConsole(p.entry.level === 'error' ? state.consoleErrors : state.consoleWarnings, { source: p.entry.source, text: p.entry.text, url: p.entry.url || null, lineNumber: p.entry.lineNumber || null });
    }
  });
  return { state, off };
}

async function evaluate(cdp, sessionId, expression) {
  const result = await cdp.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, sessionId);
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text || 'browser evaluation failed');
  return result.result.value;
}

async function settle(cdp, sessionId, network, timeoutMs) {
  const started = Date.now();
  let ready = false;
  while (Date.now() - started < timeoutMs) {
    ready = await evaluate(cdp, sessionId, `document.readyState === 'complete' && Boolean(document.body) && document.body.innerText.trim().length > 0`);
    if (ready && network.pending.size === 0 && Date.now() - network.lastActivity >= 700) break;
    await sleep(100);
  }
  const timedOut = !(ready && network.pending.size === 0 && Date.now() - network.lastActivity >= 700);
  await evaluate(cdp, sessionId, `Promise.race([document.fonts?.ready || Promise.resolve(), new Promise(r => setTimeout(r, 5000))]).then(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))))`);
  await sleep(120);
  return { elapsedMs: Date.now() - started, timedOut, pendingAtCapture: [...network.pending].map((id) => network.requests.get(id)).filter(Boolean) };
}

async function platformFonts(cdp, sessionId, records) {
  const documentNode = await cdp.send('DOM.getDocument', { depth: -1, pierce: true }, sessionId);
  for (const record of Object.values(records)) {
    if (!record) continue;
    const queried = await cdp.send('DOM.querySelector', { nodeId: documentNode.root.nodeId, selector: `[data-ux-evidence-font-id="${record.id}"]` }, sessionId);
    if (!queried.nodeId) { record.platformFonts = []; continue; }
    try { record.platformFonts = (await cdp.send('CSS.getPlatformFontsForNode', { nodeId: queried.nodeId }, sessionId)).fonts; }
    catch (error) { record.platformFonts = []; record.platformFontError = error.message; }
    delete record.id;
  }
  await evaluate(cdp, sessionId, `document.querySelectorAll('[data-ux-evidence-font-id]').forEach(node => node.removeAttribute('data-ux-evidence-font-id'))`);
}

function safeWrite(path, content) { mkdirSync(dirname(path), { recursive: true }); writeFileSync(path, content); }
function outputPath(out, ...parts) { const path = join(out, ...parts); mkdirSync(dirname(path), { recursive: true }); return path; }
function reportPath(out, path) { return relative(out, path).split('\\').join('/'); }

async function screenshotSet(cdp, sessionId, out, parts) {
  const layout = await cdp.send('Page.getLayoutMetrics', {}, sessionId);
  const size = layout.cssContentSize || layout.contentSize;
  const files = {};
  const full = outputPath(out, ...parts, 'full.png');
  const fullShot = await cdp.send('Page.captureScreenshot', { format: 'png', fromSurface: true, captureBeyondViewport: true, clip: { x: 0, y: 0, width: Math.max(1, Math.ceil(size.width)), height: Math.max(1, Math.ceil(size.height)), scale: 1 } }, sessionId);
  safeWrite(full, Buffer.from(fullShot.data, 'base64')); files.full = reportPath(out, full);
  const scrollHeight = await evaluate(cdp, sessionId, `Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)`);
  const viewportHeight = await evaluate(cdp, sessionId, `innerHeight`);
  const positions = { top: 0, middle: Math.max(0, (scrollHeight - viewportHeight) / 2), bottom: Math.max(0, scrollHeight - viewportHeight) };
  for (const [name, y] of Object.entries(positions)) {
    await evaluate(cdp, sessionId, `scrollTo(0, ${JSON.stringify(y)}); new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))`);
    const actualY = await evaluate(cdp, sessionId, 'scrollY');
    const path = outputPath(out, ...parts, `${name}.png`);
    const shot = await cdp.send('Page.captureScreenshot', { format: 'png', fromSurface: true }, sessionId);
    safeWrite(path, Buffer.from(shot.data, 'base64')); files[name] = { path: reportPath(out, path), scrollY: actualY };
  }
  return files;
}

async function capturePage(browser, options) {
  const { cdp } = browser;
  const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
  const attached = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
  const sessionId = attached.sessionId;
  const observer = observe(cdp, sessionId);
  try {
    await Promise.all(['Page.enable', 'Runtime.enable', 'Network.enable', 'Log.enable', 'DOM.enable'].map((method) => cdp.send(method, {}, sessionId)));
    await cdp.send('CSS.enable', {}, sessionId);
    await cdp.send('Network.setCacheDisabled', { cacheDisabled: true }, sessionId);
    await cdp.send('Emulation.setScrollbarsHidden', { hidden: true }, sessionId);
    await cdp.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }] }, sessionId);
    if (options.emulate !== false) await cdp.send('Emulation.setDeviceMetricsOverride', { width: options.viewport.width, height: options.viewport.height, screenWidth: options.viewport.width, screenHeight: options.viewport.height, deviceScaleFactor: 1, mobile: false }, sessionId);
    await cdp.send('Page.addScriptToEvaluateOnNewDocument', { source: localeBootstrap(options.language) }, sessionId);
    const load = new Promise((accept) => {
      const off = cdp.on((message) => { if (message.sessionId === sessionId && message.method === 'Page.loadEventFired') { off(); accept(); } });
    });
    const navigationStarted = Date.now();
    const navigation = await cdp.send('Page.navigate', { url: options.url }, sessionId);
    if (navigation.errorText) throw new Error(`navigation failed: ${navigation.errorText}`);
    await Promise.race([load, sleep(options.timeoutMs).then(() => { throw new Error('Page.loadEventFired timed out'); })]);
    const settled = await settle(cdp, sessionId, observer.state, options.timeoutMs);
    const audit = await evaluate(cdp, sessionId, PAGE_AUDIT);
    await platformFonts(cdp, sessionId, audit.fonts);
    const normalMotion = await evaluate(cdp, sessionId, MOTION_AUDIT);
    await cdp.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] }, sessionId);
    await evaluate(cdp, sessionId, `new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))`);
    const reducedMotion = await evaluate(cdp, sessionId, MOTION_AUDIT);
    await cdp.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }] }, sessionId);
    const screenshots = await screenshotSet(cdp, sessionId, options.out, options.parts);
    const apiNetworkRequests = [...observer.state.requests.values()].filter(({ url }) => { try { return new URL(url).pathname.startsWith('/api/'); } catch { return false; } });
    return {
      requestedUrl: options.url, finalUrl: audit.url, title: audit.title, locale: options.language.locale, direction: options.language.direction,
      viewport: options.viewport, page: audit, network: { httpErrors: observer.state.httpErrors, failures: observer.state.failures, cancellations: observer.state.cancellations, apiRequests: apiNetworkRequests },
      console: { errors: observer.state.consoleErrors, warnings: observer.state.consoleWarnings },
      motion: { normal: normalMotion, reduced: reducedMotion, reducedConfiguredCount: reducedMotion.configuredEffects.length, reducedMotionSensitiveResidualCount: reducedMotion.motionSensitiveEffects.length },
      load: { navigationToCaptureMs: Date.now() - navigationStarted, settle: settled, performance: audit.timing }, screenshots,
    };
  } finally {
    observer.off();
    await cdp.send('Target.closeTarget', { targetId }).catch(() => {});
  }
}

async function attemptCapture(browser, options) {
  try { return await capturePage(browser, options); }
  catch (error) {
    process.stderr.write(`capture failed: ${error.message}\n`);
    return { requestedUrl: options.url, locale: options.language.locale, direction: options.language.direction, viewport: options.viewport, error: { message: error.message, stack: error.stack } };
  }
}

async function runMatrix(options) {
  if (!options.routes.length) return [];
  const browser = await new PrivateChrome({ width: 1728, height: 900 }).start();
  const reports = [];
  try {
    for (const route of options.routes) {
      const routeReport = { slug: route.slug, address: route.address, captures: [] };
      for (const language of LOCALES) for (const viewport of VIEWPORTS) {
        process.stderr.write(`capture ${route.slug} ${language.locale} ${viewport.slug}\n`);
        routeReport.captures.push(await attemptCapture(browser, { ...options, language, viewport, url: new URL(route.address, options.baseUrl).href, parts: [route.slug, language.locale, viewport.slug] }));
        safeWrite(join(options.out, route.slug, 'report.json'), `${JSON.stringify(routeReport, null, 2)}\n`);
      }
      reports.push({ slug: route.slug, report: `${route.slug}/report.json`, failureCount: routeReport.captures.filter(({ error }) => error).length });
    }
  } finally { await browser.stop(); }
  return reports;
}

async function runGate(options) {
  if (!options.gate) return null;
  const viewport = { slug: '1024x768', width: 1024, height: 768 };
  const browser = await new PrivateChrome(viewport).start();
  const report = { slug: 'desktop-gate', address: '/', captures: [] };
  try {
    for (const language of LOCALES) {
      process.stderr.write(`capture desktop-gate ${language.locale} ${viewport.slug}\n`);
      const captured = await attemptCapture(browser, { ...options, language, viewport, url: new URL('/', options.baseUrl).href, parts: ['desktop-gate', language.locale, viewport.slug] });
      if (!captured.error) captured.gateContract = {
        exactViewport: ['innerWidth', 'outerWidth', 'screenWidth', 'availWidth'].every((key) => captured.page.viewport[key] === 1024)
          && ['innerHeight', 'outerHeight', 'screenHeight', 'availHeight'].every((key) => captured.page.viewport[key] === 768),
        apiResourceListEmpty: captured.page.apiResourceList.length === 0,
        apiNetworkRequestsEmpty: captured.network.apiRequests.length === 0,
      };
      report.captures.push(captured);
      safeWrite(join(options.out, 'desktop-gate', 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
    }
    return { slug: 'desktop-gate', report: 'desktop-gate/report.json', failureCount: report.captures.filter(({ error }) => error).length };
  } finally { await browser.stop(); }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  mkdirSync(options.out, { recursive: true });
  const generatedAt = new Date().toISOString();
  const capturedRoutes = await runMatrix(options);
  const capturedGate = await runGate(options);
  const routes = options.resume
    ? ROUTES.filter(({ slug }) => existsSync(join(options.out, slug, 'report.json'))).map(({ slug }) => ({ slug, report: `${slug}/report.json`, failureCount: readFileSync(join(options.out, slug, 'report.json'), 'utf8').includes('"error"') ? 1 : 0 }))
    : capturedRoutes;
  const existingGate = existsSync(join(options.out, 'desktop-gate', 'report.json')) ? { slug: 'desktop-gate', report: 'desktop-gate/report.json', failureCount: 0 } : null;
  const gate = capturedGate || (options.resume ? existingGate : null);
  const aggregateOptions = options.resume ? { ...options, routes: ROUTES, gate: Boolean(gate) } : options;
  const aggregate = buildAggregate(aggregateOptions, routes, gate, { locales: LOCALES, viewports: VIEWPORTS });
  safeWrite(join(options.out, 'aggregate.json'), `${JSON.stringify(aggregate, null, 2)}\n`);
  safeWrite(join(options.out, 'aggregate.md'), aggregateMarkdown(aggregate));
  const index = {
    schemaVersion: 2, generatedAt, completedAt: new Date().toISOString(), baseUrl: options.baseUrl,
    policy: { navigationOnly: true, backendWrites: false, locales: LOCALES, viewports: VIEWPORTS, localeResponseRewriteEndpoints: ['/api/overview', '/api/settings'] },
    routes, ...(gate ? { gate } : {}), aggregate: { json: 'aggregate.json', markdown: 'aggregate.md', verdict: aggregate.verdict, defectCount: aggregate.defectCount },
  };
  safeWrite(join(options.out, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
  process.stdout.write(`${join(options.out, 'index.json')}\n`);
  if (capturedRoutes.some(({ failureCount }) => failureCount) || capturedGate?.failureCount) process.exitCode = 1;
}

async function stopActiveBrowsers() { await Promise.allSettled([...ACTIVE_BROWSERS].map((browser) => browser.stop())); }
let active = true;
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => {
  if (!active) return;
  active = false;
  stopActiveBrowsers().finally(() => process.exit(130));
});
main()
  .catch((error) => { process.stderr.write(`${error.stack || error.message}\n`); process.exitCode = 1; })
  .finally(async () => { active = false; await stopActiveBrowsers(); });
