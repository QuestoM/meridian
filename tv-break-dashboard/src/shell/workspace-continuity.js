import { flushSync } from 'react-dom';

let continuityFrame = 0;
let continuityFinishers = [];

function prefersReducedMotion() {
  return typeof window !== 'undefined'
    && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}

function finishContinuityCallbacks() {
  const callbacks = continuityFinishers;
  continuityFinishers = [];
  callbacks.forEach((callback) => callback?.());
}

export function queueWorkspaceContinuity(onFinished) {
  if (onFinished) continuityFinishers.push(onFinished);
  if (continuityFrame) return;
  continuityFrame = window.requestAnimationFrame(() => {
    continuityFrame = 0;
    if (document.documentElement.dataset.shellViewTransition === 'active' || prefersReducedMotion()) {
      finishContinuityCallbacks();
      return;
    }

    const root = document.querySelector(
      '#kairos-main > .page-workspace, #kairos-main > .rules-workspace',
    );
    const visiblePanels = root
      ? Array.from(root.querySelectorAll('[role="tabpanel"]')).filter((panel) => (
        panel.getAttribute('aria-hidden') !== 'true'
        && !panel.hidden
        && panel.getClientRects().length > 0
      ))
      : [];
    const target = visiblePanels.at(-1) || root;
    if (!target) {
      finishContinuityCallbacks();
      return;
    }

    target.classList.remove('shell-continuity-in');
    void target.offsetWidth;
    target.classList.add('shell-continuity-in');
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      target.classList.remove('shell-continuity-in');
      finishContinuityCallbacks();
    };
    target.addEventListener('animationend', finish, { once: true });
    window.setTimeout(finish, 320);
  });
}

export function transitionWorkspaceUpdate(update, { focusMain = false } = {}) {
  const focus = () => {
    if (focusMain) document.getElementById('kairos-main')?.focus({ preventScroll: true });
  };
  if (!document.startViewTransition || prefersReducedMotion()) {
    update();
    queueWorkspaceContinuity(focus);
    return;
  }

  document.documentElement.dataset.shellViewTransition = 'active';
  let committed = false;
  const commit = () => {
    if (committed) return;
    committed = true;
    flushSync(update);
  };
  let transition;
  try {
    transition = document.startViewTransition(commit);
  } catch {
    delete document.documentElement.dataset.shellViewTransition;
    commit();
    queueWorkspaceContinuity(focus);
    return;
  }
  const finish = () => {
    delete document.documentElement.dataset.shellViewTransition;
    focus();
  };
  transition.updateCallbackDone.catch(() => {
    if (!committed) commit();
  });
  transition.finished.then(finish, finish);
}
