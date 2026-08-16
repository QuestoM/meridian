// Browser-evaluated, serializable probes kept separate from CDP orchestration.
export const PAGE_AUDIT = `(() => {
  const text = (value, limit = 160) => String(value || '').replace(/\\s+/g, ' ').trim().slice(0, limit);
  const rect = (el) => { const r = el.getBoundingClientRect(); return { x: +r.x.toFixed(2), y: +r.y.toFixed(2), width: +r.width.toFixed(2), height: +r.height.toFixed(2) }; };
  const shown = (el, allowTransparent = false) => {
    if (!el || !el.isConnected || el.closest('[hidden],[inert]')) return false;
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    return s.display !== 'none' && s.visibility !== 'hidden' && s.visibility !== 'collapse' && (allowTransparent || Number(s.opacity) > 0) && r.width > 0 && r.height > 0;
  };
  const selector = (el) => {
    if (el.id) return '#' + CSS.escape(el.id);
    const testId = el.getAttribute('data-testid'); if (testId) return '[data-testid="' + CSS.escape(testId) + '"]';
    const label = el.getAttribute('aria-label'); if (label) return el.tagName.toLowerCase() + '[aria-label="' + CSS.escape(label) + '"]';
    const names = [...el.classList].slice(0, 2).map((name) => '.' + CSS.escape(name)).join('');
    return el.tagName.toLowerCase() + names;
  };
  const visibility = (el) => {
    const style = getComputedStyle(el), r = el.getBoundingClientRect();
    const disabledAncestor = el.closest(':disabled,[aria-disabled="true"]');
    return {
      connected: el.isConnected, display: style.display, visibility: style.visibility,
      opacity: +style.opacity, pointerEvents: style.pointerEvents, cursor: style.cursor,
      clientRectCount: el.getClientRects().length,
      intersectsViewport: r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth,
      disabled: Boolean(disabledAncestor), disabledSelector: disabledAncestor ? selector(disabledAncestor) : null,
      tabIndex: el.tabIndex, focusable: el.matches('a[href],button,input,select,textarea,summary,[tabindex]') && !disabledAncestor,
    };
  };
  const paint = (el) => {
    const style = getComputedStyle(el);
    return { color: style.color, backgroundColor: style.backgroundColor, borderColor: style.borderColor, outlineColor: style.outlineColor };
  };
  const wrapperEvidence = (node) => ({ selector: selector(node), rect: rect(node), visibility: visibility(node), computedPaint: paint(node) });
  const candidates = [...document.querySelectorAll('a[href],button,input:not([type="hidden"]),select,textarea,summary,[role="button"],[role="link"],[role="tab"],[role="checkbox"],[role="radio"],[role="switch"],[tabindex]:not([tabindex="-1"])')]
    .filter((el, index, all) => all.indexOf(el) === index && !el.disabled && el.getAttribute('aria-disabled') !== 'true');
  const targetFailures = [], nativeWrapperExceptions = [];
  for (const el of candidates) {
    const native = el.matches('input,select,textarea');
    const isShown = shown(el, native);
    if (!isShown) continue;
    const own = rect(el); if (own.width + .1 >= 44 && own.height + .1 >= 44) continue;
    const linked = el.id ? document.querySelector('label[for="' + CSS.escape(el.id) + '"]') : null;
    const wrappers = [...new Set([linked, el.closest('label'), el.closest('.MuiButtonBase-root,.MuiInputBase-root,.MuiSlider-root,.MuiSwitch-root,[data-hit-target]')].filter(Boolean))];
    const wrapper = wrappers.find((node) => shown(node) && node.getBoundingClientRect().width + .1 >= 44 && node.getBoundingClientRect().height + .1 >= 44);
    const record = {
      selector: selector(el), tag: el.tagName.toLowerCase(), type: el.getAttribute('type'),
      role: el.getAttribute('role'), label: text(el.getAttribute('aria-label') || el.innerText || el.value),
      text: text(el.innerText || el.value || el.getAttribute('aria-label')), rect: own,
      visibility: visibility(el), computedPaint: paint(el), wrapperCandidates: wrappers.map(wrapperEvidence),
    };
    if (native && wrapper) nativeWrapperExceptions.push({ ...record, exception: 'native control delegates its hit area to an explicit wrapper at least 44x44', wrapper: { selector: selector(wrapper), rect: rect(wrapper) } });
    else targetFailures.push(record);
  }
  const parse = (value) => {
    const match = String(value).match(/rgba?\\(\\s*([\\d.]+)[, ]+([\\d.]+)[, ]+([\\d.]+)(?:\\s*[,/]\\s*([\\d.]+))?\\s*\\)/i);
    return match ? [+match[1], +match[2], +match[3], match[4] === undefined ? 1 : +match[4]] : null;
  };
  const over = (top, bottom) => { const a = top[3] + bottom[3] * (1 - top[3]); return a ? [(top[0]*top[3]+bottom[0]*bottom[3]*(1-top[3]))/a,(top[1]*top[3]+bottom[1]*bottom[3]*(1-top[3]))/a,(top[2]*top[3]+bottom[2]*bottom[3]*(1-top[3]))/a,a] : [0,0,0,0]; };
  const luminance = (rgb) => { const c = rgb.slice(0,3).map((v) => { v /= 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4; }); return .2126*c[0]+.7152*c[1]+.0722*c[2]; };
  const contrast = (a, b) => { const x = luminance(a), y = luminance(b); return (Math.max(x,y)+.05)/(Math.min(x,y)+.05); };
  const directText = [...document.querySelectorAll('body *')].filter((el) => shown(el) && [...el.childNodes].some((node) => node.nodeType === Node.TEXT_NODE && text(node.textContent)));
  const contrastFailures = [], inactiveExceptions = [], skippedContrast = []; let contrastChecked = 0;
  for (const el of directText) {
    const sample = text([...el.childNodes].filter((n) => n.nodeType === Node.TEXT_NODE).map((n) => n.textContent).join(' '));
    if (!sample) continue;
    const chain = []; for (let node = el; node; node = node.parentElement) chain.unshift(node);
    let bg = [255,255,255,1], uncertain = false;
    for (const node of chain) {
      const style = getComputedStyle(node), parsed = parse(style.backgroundColor);
      if (parsed) { bg = over(parsed, bg); if (parsed[3] >= .999) uncertain = false; }
      if (style.backgroundImage !== 'none') uncertain = true;
    }
    const style = getComputedStyle(el), fg = parse(style.color);
    if (!fg || uncertain) { skippedContrast.push({ selector: selector(el), text: sample, reason: uncertain ? 'background image or gradient' : 'unparsed computed color' }); continue; }
    const painted = over(fg, bg), size = parseFloat(style.fontSize), weight = parseInt(style.fontWeight, 10) || (style.fontWeight === 'bold' ? 700 : 400);
    const threshold = size >= 24 || (size >= 18.66 && weight >= 700) ? 3 : 4.5;
    const ratio = contrast(painted, bg); contrastChecked += 1;
    if (ratio + .005 < threshold && contrastFailures.length + inactiveExceptions.length < 250) {
      const record = {
        selector: selector(el), text: sample, rect: rect(el), visibility: visibility(el),
        foreground: style.color, background: 'rgb(' + bg.slice(0,3).map(Math.round).join(', ') + ')',
        ratio: +ratio.toFixed(2), threshold, fontSize: size, fontWeight: weight,
      };
      if (record.visibility.disabled) inactiveExceptions.push({ ...record, exception: 'WCAG text contrast does not apply to an inactive UI component' });
      else contrastFailures.push(record);
    }
  }
  const fontCandidates = directText.map((el) => ({ el, value: text(el.textContent) }));
  const definitions = { hebrew: /[\\u0590-\\u05ff]/, latin: /[A-Za-z]/, figure: /[0-9]/ }, fonts = {};
  for (const [kind, pattern] of Object.entries(definitions)) {
    const found = fontCandidates.find(({ value }) => pattern.test(value));
    if (!found) { fonts[kind] = null; continue; }
    const id = 'ux-font-' + kind, style = getComputedStyle(found.el); found.el.setAttribute('data-ux-evidence-font-id', id);
    fonts[kind] = { id, selector: selector(found.el), text: found.value.slice(0,80), computedFamily: style.fontFamily, fontSize: style.fontSize, fontWeight: style.fontWeight, direction: style.direction };
  }
  const root = document.documentElement, body = document.body;
  const overflowOffenders = [...document.querySelectorAll('body *')].filter((el) => shown(el) && (el.getBoundingClientRect().right > root.clientWidth + 1 || el.getBoundingClientRect().left < -1)).slice(0,100).map((el) => ({ selector: selector(el), rect: rect(el) }));
  const measured = (query) => [...document.querySelectorAll(query)].filter(shown).map((el) => ({ selector: selector(el), rect: rect(el) }));
  const main = document.querySelector('#kairos-main') || document.querySelector('main') || body;
  const topBar = document.querySelector('.top-bar');
  const routeRoot = [...main.children].find((el) => shown(el) && el.matches('.page-workspace,.rules-workspace,.mc-body')) || main.querySelector('.page-workspace,.rules-workspace,.mc-body') || main;
  const box = (el) => {
    const style = getComputedStyle(el), r = el.getBoundingClientRect();
    return { selector: selector(el), rect: rect(el), padding: { top: parseFloat(style.paddingTop) || 0, right: parseFloat(style.paddingRight) || 0, bottom: parseFloat(style.paddingBottom) || 0, left: parseFloat(style.paddingLeft) || 0 } };
  };
  const mainRect = main.getBoundingClientRect(), topRect = topBar?.getBoundingClientRect();
  const mainViewport = { left: Math.max(0, mainRect.left), right: Math.min(innerWidth, mainRect.right), top: Math.max(0, mainRect.top, topRect?.bottom || 0), bottom: Math.min(innerHeight, mainRect.bottom) };
  const scrollPosition = { x: scrollX, y: scrollY, maxX: Math.max(0, root.scrollWidth - root.clientWidth), maxY: Math.max(0, root.scrollHeight - root.clientHeight) };
  const rtl = getComputedStyle(routeRoot).direction === 'rtl';
  const logicalInsets = (r) => ({ blockStart: r.top - mainViewport.top, blockEnd: mainViewport.bottom - r.bottom, inlineStart: rtl ? mainViewport.right - r.right : r.left - mainViewport.left, inlineEnd: rtl ? r.left - mainViewport.left : mainViewport.right - r.right });
  const fullBleedAllowlist = [
    { selector: '.day-board,.timeline-scroll,.transmission-ribbon', reason: 'broadcast timeline instrument' },
    { selector: '.plan-board-stage,.planning-canvas-timeline,.timeline-view', reason: 'planning timeline instrument' },
    { selector: '.break-board,.pod-spot-table,.pod-rate-table', reason: 'traffic board or pod table' },
    { selector: '.MuiDataGrid-root,.data-table-shell,.rows-drawer-table,[role="table"],table', reason: 'tabular data viewport' },
  ];
  const sectionCandidates = [...routeRoot.children].filter((el) => shown(el) && !el.matches('script,style,.toast,[role="dialog"]') && el.getBoundingClientRect().width > 8 && el.getBoundingClientRect().height > 8);
  const sectionInsets = [], edgeFailures = [], allowedEdgeContacts = [];
  for (const el of sectionCandidates) {
    const r = el.getBoundingClientRect(), insets = logicalInsets(r), record = { selector: selector(el), rect: rect(el), insets: Object.fromEntries(Object.entries(insets).map(([key,value]) => [key,+value.toFixed(2)])) };
    const meaningful = {
      blockStart: scrollPosition.y <= 1 && r.top >= mainViewport.top - 1 && r.top < mainViewport.bottom,
      blockEnd: scrollPosition.y >= scrollPosition.maxY - 1 && r.bottom <= mainViewport.bottom + 1 && r.bottom > mainViewport.top,
      inlineStart: r.right > mainViewport.left && r.left < mainViewport.right,
      inlineEnd: r.right > mainViewport.left && r.left < mainViewport.right,
    };
    const touched = Object.keys(insets).filter((edge) => meaningful[edge] && insets[edge] < 12);
    const allowance = fullBleedAllowlist.find((item) => el.matches(item.selector));
    sectionInsets.push({ ...record, meaningful, touchedEdges: touched });
    if (touched.length && allowance) allowedEdgeContacts.push({ ...record, touchedEdges: touched, allowlistSelector: allowance.selector, reason: allowance.reason });
    else if (touched.length) edgeFailures.push({ ...record, touchedEdges: touched });
  }
  const h1 = [...document.querySelectorAll('h1')];
  const navigation = performance.getEntriesByType('navigation')[0], fcp = performance.getEntriesByName('first-contentful-paint')[0];
  return {
    url: location.href, title: document.title,
    locale: { requestedStorage: localStorage.getItem('kairos.locale'), lang: root.lang, dir: root.dir, computedDirection: getComputedStyle(body).direction },
    h1: { count: h1.length, visibleCount: h1.filter((node) => shown(node)).length, text: h1.map((node) => text(node.textContent)) },
    viewport: { innerWidth, innerHeight, outerWidth, outerHeight, screenWidth: screen.width, screenHeight: screen.height, availWidth: screen.availWidth, availHeight: screen.availHeight, devicePixelRatio },
    document: { clientWidth: root.clientWidth, clientHeight: root.clientHeight, scrollWidth: Math.max(root.scrollWidth, body.scrollWidth), scrollHeight: Math.max(root.scrollHeight, body.scrollHeight), scrollPosition, horizontalOverflow: Math.max(root.scrollWidth, body.scrollWidth) > root.clientWidth + 1, overflowOffenders },
    layout: {
      shellHeights: { topBar: measured('.top-bar'), topBarPrimary: measured('.top-bar-primary'), shellLocalNav: measured('.context-local-nav'), workspaceLocalNav: measured('.studio-local-nav'), routeHeader: measured('.page-header,.studio-context-header,.mc-header') },
      main: box(main), routeRoot: box(routeRoot), mainViewport: Object.fromEntries(Object.entries(mainViewport).map(([key,value]) => [key,+value.toFixed(2)])),
      routeRootInsets: Object.fromEntries(Object.entries(logicalInsets(routeRoot.getBoundingClientRect())).map(([key,value]) => [key,+value.toFixed(2)])),
      edgeRule: { minimumLogicalInsetCssPixels: 12, blockEdgesMeasuredOnlyAtMatchingDocumentBoundary: true, fullBleedAllowlist }, sectionInsets, edgeFailures, allowedEdgeContacts,
    },
    interactiveTargets: { minimumCssPixels: 44, failures: targetFailures, nativeWrapperExceptions },
    contrast: { standard: 'WCAG AA text thresholds', checked: contrastChecked, failures: contrastFailures, inactiveExceptions, skipped: skippedContrast.slice(0,100) }, fonts,
    timing: navigation ? { type: navigation.type, responseEndMs: +navigation.responseEnd.toFixed(1), domContentLoadedMs: +navigation.domContentLoadedEventEnd.toFixed(1), loadMs: +navigation.loadEventEnd.toFixed(1), durationMs: +navigation.duration.toFixed(1), firstContentfulPaintMs: fcp ? +fcp.startTime.toFixed(1) : null } : null,
    apiResourceList: performance.getEntriesByType('resource').map((entry) => entry.name).filter((name) => { try { return new URL(name).pathname.startsWith('/api/'); } catch { return false; } }),
  };
})()`;

export const MOTION_AUDIT = `(() => {
  const shown = (el) => { const s = getComputedStyle(el), r = el.getBoundingClientRect(); return s.display !== 'none' && s.visibility !== 'hidden' && Number(s.opacity) > 0 && r.width > 0 && r.height > 0; };
  const selector = (el) => el.id ? '#' + CSS.escape(el.id) : el.tagName.toLowerCase() + [...el.classList].slice(0,2).map((name) => '.' + CSS.escape(name)).join('');
  const milliseconds = (list) => Math.max(0, ...String(list).split(',').map((part) => { const value = parseFloat(part) || 0; return part.trim().endsWith('ms') ? value : value * 1000; }));
  const active = [];
  for (const el of [...document.querySelectorAll('body *')].filter(shown)) {
    const style = getComputedStyle(el), transitionMs = milliseconds(style.transitionDuration), animationMs = milliseconds(style.animationDuration);
    if ((style.transitionProperty !== 'none' && transitionMs > .1) || (style.animationName !== 'none' && animationMs > .1)) active.push({ selector: selector(el), transitionProperty: style.transitionProperty, transitionDurationMs: transitionMs, animationName: style.animationName, animationDurationMs: animationMs, animationIterationCount: style.animationIterationCount });
    if (active.length >= 200) break;
  }
  let reducedRules = 0, viewTransitionRules = 0; const continuityRules = [];
  const inspect = (rules) => { for (const rule of [...(rules || [])]) { const css = rule.cssText || ''; if (css.includes('prefers-reduced-motion')) reducedRules += 1; if (css.includes('view-transition')) viewTransitionRules += 1; if (continuityRules.length < 30 && /shell-continuity|view-transition|prefers-reduced-motion/.test(css)) continuityRules.push(css.replace(/\\s+/g,' ').slice(0,800)); try { if (rule.cssRules) inspect(rule.cssRules); } catch {} } };
  for (const sheet of [...document.styleSheets]) { try { inspect(sheet.cssRules); } catch {} }
  const motionSensitive = active.filter((item) => item.animationName !== 'none' || /(^|,|\\s)(all|transform|translate|rotate|scale|opacity|top|right|bottom|left|inset|width|height|inline-size|block-size)(,|\\s|$)/.test(item.transitionProperty));
  return { probeScope: 'computed effects and emulated media only; no navigation control was clicked', prefersReducedMotion: matchMedia('(prefers-reduced-motion: reduce)').matches, startViewTransitionAvailable: typeof document.startViewTransition === 'function', reducedMotionRuleCount: reducedRules, viewTransitionRuleCount: viewTransitionRules, continuityRules, configuredEffects: active, motionSensitiveEffects: motionSensitive };
})()`;
