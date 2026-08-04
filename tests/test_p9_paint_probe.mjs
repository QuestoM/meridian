// Lay a document out in a real browser and report where the glyphs landed.
//
// A bidirectional defect is invisible to every runner in this repository: the
// string is correct and only the paint is wrong, and resolving the paint means
// running the Unicode bidirectional algorithm, which only a layout engine has.
// This opens a private headless Chrome over the DevTools protocol, optionally
// injects a boot script before the document runs, loads the document, evaluates
// the expression it is given and prints the result as JSON. It owns its own
// profile directory and an ephemeral debugging port, so it cannot collide with
// a browser anybody else is driving.
//
// Usage: node test_p9_paint_probe.mjs <document.html|url> <expression.js> [boot.js]
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CANDIDATES = [
  process.env.CHROME_PATH,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);

const binary = CANDIDATES.find((path) => existsSync(path));
if (!binary) {
  process.stderr.write('no chrome');
  process.exit(2);
}

const [target, expressionPath, bootPath] = process.argv.slice(2);
const url = /^https?:\/\//.test(target) ? target : `file://${target}`;
const profile = mkdtempSync(join(tmpdir(), 'p9-paint-'));
const chrome = spawn(binary, [
  '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--disable-gpu', '--hide-scrollbars',
  '--window-size=1440,900', 'about:blank',
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function browserSocket() {
  const marker = join(profile, 'DevToolsActivePort');
  for (let attempt = 0; attempt < 200; attempt += 1) {
    if (existsSync(marker)) {
      const [port, path] = readFileSync(marker, 'utf8').trim().split('\n');
      if (port && path) return `ws://127.0.0.1:${port}${path}`;
    }
    await sleep(100);
  }
  throw new Error('chrome never published a debugging port');
}

function protocol(socket) {
  let counter = 0;
  const pending = new Map();
  const watchers = [];
  socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.id !== undefined && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) reject(new Error(JSON.stringify(message.error)));
      else resolve(message.result);
    } else {
      watchers.forEach((watcher) => watcher(message));
    }
  });
  return {
    send(method, params, sessionId) {
      counter += 1;
      const id = counter;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        socket.send(JSON.stringify({ id, method, params: params || {}, sessionId }));
      });
    },
    watch(watcher) { watchers.push(watcher); },
  };
}

const socket = new WebSocket(await browserSocket());
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve);
  socket.addEventListener('error', reject);
});
const cdp = protocol(socket);
const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
await cdp.send('Page.enable', {}, sessionId);
if (bootPath) {
  await cdp.send('Page.addScriptToEvaluateOnNewDocument',
    { source: readFileSync(bootPath, 'utf8') }, sessionId);
}
const loaded = new Promise((resolve) => {
  cdp.watch((message) => { if (message.method === 'Page.loadEventFired') resolve(); });
});
await cdp.send('Page.navigate', { url }, sessionId);
await loaded;

const evaluated = await cdp.send('Runtime.evaluate', {
  expression: readFileSync(expressionPath, 'utf8'),
  awaitPromise: true,
  returnByValue: true,
}, sessionId);

if (evaluated.exceptionDetails) {
  process.stderr.write(JSON.stringify(evaluated.exceptionDetails));
  process.exitCode = 1;
} else {
  process.stdout.write(JSON.stringify(evaluated.result.value));
}
socket.close();
chrome.kill('SIGTERM');
