const allowedDirections = new Set(['master', 'frame', 'atlas']);
const requestedDirection = new URLSearchParams(window.location.search).get('direction');
const direction = allowedDirections.has(requestedDirection) ? requestedDirection : 'master';

document.body.dataset.direction = direction;
document.querySelector(`[data-direction-link="${direction}"]`)?.setAttribute('aria-current', 'page');

const directionNames = {
  master: 'Master Control',
  frame: 'Signal Frame',
  atlas: 'Network Atlas',
};
document.title = `Kairos — ${directionNames[direction]}`;

const timecode = document.querySelector('.timecode');
function updateTimecode() {
  const now = new Date();
  const frames = String(Math.floor((now.getMilliseconds() / 1000) * 25)).padStart(2, '0');
  timecode.textContent = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}:${frames}`;
}
updateTimecode();
const clock = window.setInterval(updateTimecode, 40);

const drawer = document.querySelector('.decision-drawer');
const openers = document.querySelectorAll('[data-open-decision]');
const closer = document.querySelector('.drawer-close');
let lastOpener = null;

function setDrawer(open) {
  drawer.classList.toggle('open', open);
  drawer.setAttribute('aria-hidden', String(!open));
  document.body.classList.toggle('drawer-open', open);
  if (open) window.setTimeout(() => closer.focus(), 220);
  else lastOpener?.focus();
}

openers.forEach((opener) => opener.addEventListener('click', () => {
  lastOpener = opener;
  setDrawer(true);
}));
closer.addEventListener('click', () => setDrawer(false));
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && drawer.classList.contains('open')) setDrawer(false);
});

let duration = 142;
function renderDuration() {
  document.querySelectorAll('[data-duration]').forEach((node) => { node.textContent = String(duration); });
}
document.querySelector('[data-minus]').addEventListener('click', () => {
  duration = Math.max(120, duration - 5);
  renderDuration();
});
document.querySelector('[data-plus]').addEventListener('click', () => {
  duration = Math.min(180, duration + 5);
  renderDuration();
});

window.addEventListener('pagehide', () => window.clearInterval(clock), { once: true });
