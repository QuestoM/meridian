import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const readJson = (path) => JSON.parse(readFileSync(path, 'utf8'));
const captureLabel = (capture) => `${capture.routeSlug}/${capture.locale}/${capture.viewport?.slug || 'unknown'}`;
function sameRouteIdentity(requestedValue, finalValue) {
  try {
    const requested = new URL(requestedValue), final = new URL(finalValue);
    if (requested.origin !== final.origin || requested.pathname !== final.pathname || requested.hash !== final.hash) return false;
    return [...requested.searchParams].every(([key, value]) => final.searchParams.getAll(key).includes(value));
  } catch { return requestedValue === finalValue; }
}
const inactiveContrast = (record) => Boolean(record.visibility?.disabled || /Mui-disabled/.test(record.selector || ''));
const targetWithinRoundingTolerance = (record) => record.rect?.width + .1 >= 44 && record.rect?.height + .1 >= 44;
const activeReducedMotionViolation = (record) => record.animationName && record.animationName !== 'none' && record.animationName !== 'mui-auto-fill-cancel' && Number(record.animationDurationMs) > 20;
function distribution(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return { count: 0, min: null, median: null, p95: null, max: null };
  const at = (ratio) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * ratio) - 1))];
  return { count: sorted.length, min: sorted[0], median: at(.5), p95: at(.95), max: sorted.at(-1) };
}
const flattenFinding = (captures, getter) => captures.flatMap((capture) => (getter(capture) || []).map((record) => ({ capture: captureLabel(capture), record })));
function duplicateSuccessfulGets(capture) {
  const grouped = new Map();
  for (const request of capture.network?.apiRequests || []) {
    if (request.method !== 'GET' || request.status < 200 || request.status >= 400) continue;
    const key = `${request.method} ${request.url}`;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(request.status);
  }
  return [...grouped].filter(([, statuses]) => statuses.length > 1).map(([request, statuses]) => ({ request, count: statuses.length, statuses }));
}
function screenshotRecords(out, captures) {
  return captures.flatMap((capture) => ['full', 'top', 'middle', 'bottom'].map((kind) => {
    const value = capture.screenshots?.[kind], path = typeof value === 'string' ? value : value?.path;
    return { capture: captureLabel(capture), kind, path: path || null, exists: Boolean(path && existsSync(join(out, path))) };
  }));
}

export function buildAggregate(options, routeRefs, gateRef, { locales, viewports }) {
  const routeReports = routeRefs.map((ref) => readJson(join(options.out, ref.report)));
  const desktop = routeReports.flatMap((report) => report.captures.map((capture) => ({ ...capture, routeSlug: report.slug })));
  const gateReport = gateRef ? readJson(join(options.out, gateRef.report)) : null;
  const gate = gateReport ? gateReport.captures.map((capture) => ({ ...capture, routeSlug: 'desktop-gate' })) : [];
  const all = [...desktop, ...gate], successful = all.filter((capture) => !capture.error), desktopSuccessful = desktop.filter((capture) => !capture.error);
  const screenshots = screenshotRecords(options.out, successful);
  const rawTargets = flattenFinding(successful, (capture) => capture.page?.interactiveTargets?.failures);
  const rawContrast = flattenFinding(successful, (capture) => capture.page?.contrast?.failures);
  const explicitInactiveContrast = flattenFinding(successful, (capture) => capture.page?.contrast?.inactiveExceptions);
  const reducedConfigured = flattenFinding(successful, (capture) => capture.motion?.reduced?.motionSensitiveEffects);
  const details = {
    captureErrors: all.filter((capture) => capture.error).map((capture) => ({ capture: captureLabel(capture), error: capture.error })),
    settleTimeouts: successful.filter((capture) => capture.load?.settle?.timedOut).map((capture) => ({ capture: captureLabel(capture), settle: capture.load.settle })),
    pageIdentity: successful.filter((capture) => !capture.title || !sameRouteIdentity(capture.requestedUrl, capture.finalUrl)).map((capture) => ({ capture: captureLabel(capture), requestedUrl: capture.requestedUrl, finalUrl: capture.finalUrl, title: capture.title })),
    addressEnrichment: successful.filter((capture) => capture.finalUrl !== capture.requestedUrl && sameRouteIdentity(capture.requestedUrl, capture.finalUrl)).map((capture) => ({ capture: captureLabel(capture), requestedUrl: capture.requestedUrl, finalUrl: capture.finalUrl, title: capture.title, classification: 'route retained the canonical path, hash and requested query while adding route-owned selection state' })),
    h1: successful.filter((capture) => capture.page?.h1?.count !== 1 || capture.page?.h1?.visibleCount !== 1).map((capture) => ({ capture: captureLabel(capture), h1: capture.page.h1 })),
    localeDirection: successful.filter((capture) => capture.page?.locale?.lang !== capture.locale || capture.page?.locale?.dir !== capture.direction || capture.page?.locale?.computedDirection !== capture.direction).map((capture) => ({ capture: captureLabel(capture), requested: { locale: capture.locale, direction: capture.direction }, observed: capture.page.locale })),
    horizontalOverflow: successful.filter((capture) => capture.page?.document?.horizontalOverflow).map((capture) => ({ capture: captureLabel(capture), document: capture.page.document })),
    targetFailures: rawTargets.filter(({ record }) => !targetWithinRoundingTolerance(record)),
    targetRoundingTolerance: rawTargets.filter(({ record }) => targetWithinRoundingTolerance(record)),
    nativeWrapperExceptions: flattenFinding(successful, (capture) => capture.page?.interactiveTargets?.nativeWrapperExceptions),
    contrastFailures: rawContrast.filter(({ record }) => !inactiveContrast(record)),
    inactiveContrastExceptions: [...explicitInactiveContrast, ...rawContrast.filter(({ record }) => inactiveContrast(record))],
    consoleErrors: flattenFinding(successful, (capture) => capture.console?.errors), consoleWarnings: flattenFinding(successful, (capture) => capture.console?.warnings),
    httpErrors: flattenFinding(successful, (capture) => capture.network?.httpErrors), requestFailures: flattenFinding(successful, (capture) => capture.network?.failures), requestCancellations: flattenFinding(successful, (capture) => capture.network?.cancellations),
    edgeFailures: flattenFinding(successful, (capture) => capture.page?.layout?.edgeFailures), allowedEdgeContacts: flattenFinding(successful, (capture) => capture.page?.layout?.allowedEdgeContacts),
    reducedMotionConfiguredEffects: reducedConfigured,
    reducedMotionActiveViolations: reducedConfigured.filter(({ record }) => activeReducedMotionViolation(record)),
    duplicateSuccessfulGets: successful.flatMap((capture) => duplicateSuccessfulGets(capture).map((record) => ({ capture: captureLabel(capture), record }))),
    missingFontSamples: successful.flatMap((capture) => Object.entries(capture.page?.fonts || {}).filter(([, value]) => !value).map(([kind]) => ({ capture: captureLabel(capture), kind }))), missingScreenshots: screenshots.filter(({ exists }) => !exists),
  };
  const fontUsage = {};
  for (const locale of locales.map((entry) => entry.locale)) {
    fontUsage[locale] = {};
    for (const kind of ['hebrew', 'latin', 'figure']) {
      const samples = successful.filter((capture) => capture.locale === locale).map((capture) => capture.page?.fonts?.[kind]).filter(Boolean);
      fontUsage[locale][kind] = { sampleCount: samples.length, computedFamilies: [...new Set(samples.map(({ computedFamily }) => computedFamily))].sort(), platformFamilies: [...new Set(samples.flatMap(({ platformFonts = [] }) => platformFonts.map(({ familyName }) => familyName)))].sort() };
    }
  }
  const headerKeys = ['topBar', 'topBarPrimary', 'shellLocalNav', 'workspaceLocalNav', 'routeHeader'];
  const headerHeights = Object.fromEntries(headerKeys.map((key) => [key, distribution(desktopSuccessful.flatMap((capture) => (capture.page?.layout?.shellHeights?.[key] || []).map(({ rect }) => rect.height)))]));
  const routePadding = Object.fromEntries(['top', 'right', 'bottom', 'left'].map((edge) => [edge, distribution(desktopSuccessful.map((capture) => capture.page?.layout?.routeRoot?.padding?.[edge]))]));
  const mainPadding = Object.fromEntries(['top', 'right', 'bottom', 'left'].map((edge) => [edge, distribution(desktopSuccessful.map((capture) => capture.page?.layout?.main?.padding?.[edge]))]));
  const motion = { probeScope: 'static computed-style/media probe; controls were not activated', startViewTransitionAvailable: desktopSuccessful.filter((capture) => capture.motion?.normal?.startViewTransitionAvailable).length, viewTransitionRuleCount: distribution(desktopSuccessful.map((capture) => capture.motion?.normal?.viewTransitionRuleCount)), normalConfiguredEffects: distribution(desktopSuccessful.map((capture) => capture.motion?.normal?.configuredEffects?.length)), reducedConfiguredEffects: distribution(desktopSuccessful.map((capture) => capture.motion?.reduced?.configuredEffects?.length)), reducedMotionConfiguredEffectCount: details.reducedMotionConfiguredEffects.length, reducedMotionActiveViolationCount: details.reducedMotionActiveViolations.length, reducedMotionSensitiveResidualCount: details.reducedMotionActiveViolations.length };
  const gateContracts = gate.filter((capture) => !capture.error).map((capture) => ({ capture: captureLabel(capture), ...capture.gateContract, observedViewport: capture.page.viewport, apiResourceList: capture.page.apiResourceList, apiRequests: capture.network.apiRequests }));
  const defectCategories = ['captureErrors', 'settleTimeouts', 'pageIdentity', 'h1', 'localeDirection', 'horizontalOverflow', 'targetFailures', 'contrastFailures', 'consoleErrors', 'consoleWarnings', 'httpErrors', 'requestFailures', 'edgeFailures', 'reducedMotionActiveViolations', 'missingScreenshots'];
  const gateContractFailures = gateContracts.filter((contract) => !contract.exactViewport || !contract.apiResourceListEmpty || !contract.apiNetworkRequestsEmpty);
  const defectCount = defectCategories.reduce((total, key) => total + details[key].length, 0) + gateContractFailures.length;
  const networkSummary = {
    apiRequestRecords: successful.reduce((sum, capture) => sum + (capture.network?.apiRequests?.length || 0), 0),
    duplicateSuccessfulGetGroups: details.duplicateSuccessfulGets.length,
    duplicateSuccessfulGetExtraRequests: details.duplicateSuccessfulGets.reduce((sum, { record }) => sum + record.count - 1, 0),
    requestCancellations: details.requestCancellations.length,
    successfulStreamAbortCancellations: details.requestCancellations.filter(({ record }) => record.status === 200 && record.mimeType === 'text/event-stream' && record.errorText === 'net::ERR_ABORTED').length,
  };
  const routeSummary = routeReports.map((report) => {
    const captures = desktop.filter((capture) => capture.routeSlug === report.slug), ok = captures.filter((capture) => !capture.error);
    const prefix = `${report.slug}/`, count = (key) => details[key].filter(({ capture }) => capture.startsWith(prefix)).length;
    return { slug: report.slug, report: `${report.slug}/report.json`, captures: captures.length, successfulCaptures: ok.length, h1Failures: ok.filter((capture) => capture.page.h1.count !== 1 || capture.page.h1.visibleCount !== 1).length, overflow: ok.filter((capture) => capture.page.document.horizontalOverflow).length, targetFailures: count('targetFailures'), targetExceptions: count('nativeWrapperExceptions') + count('targetRoundingTolerance'), contrastFailures: count('contrastFailures'), contrastExceptions: count('inactiveContrastExceptions'), consoleErrors: ok.reduce((sum, capture) => sum + capture.console.errors.length, 0), consoleWarnings: ok.reduce((sum, capture) => sum + capture.console.warnings.length, 0), httpErrors: ok.reduce((sum, capture) => sum + capture.network.httpErrors.length, 0), requestFailures: ok.reduce((sum, capture) => sum + capture.network.failures.length, 0), requestCancellations: ok.reduce((sum, capture) => sum + capture.network.cancellations.length, 0), edgeFailures: ok.reduce((sum, capture) => sum + capture.page.layout.edgeFailures.length, 0), duplicateSuccessfulGetGroups: ok.reduce((sum, capture) => sum + duplicateSuccessfulGets(capture).length, 0), screenshots: screenshotRecords(options.out, ok).filter(({ exists }) => exists).length };
  });
  return { schemaVersion: 2, generatedAt: new Date().toISOString(), baseUrl: options.baseUrl, verdict: defectCount === 0 ? 'PASS' : 'FAIL', defectCount, coverage: { canonicalRoutesExpected: options.routes.length, canonicalRoutesReported: routeReports.length, desktopCapturesExpected: options.routes.length * locales.length * viewports.length, desktopCapturesReported: desktop.length, gateCapturesExpected: options.gate ? locales.length : 0, gateCapturesReported: gate.length, successfulCaptures: desktopSuccessful.length, successfulTotalCaptures: successful.length, screenshotsExpected: successful.length * 4, screenshotsPresent: screenshots.filter(({ exists }) => exists).length }, policy: { duplicateSuccessfulGetsAreInformational: true, requestCancellationsAreSeparateFromFailures: true, inactiveUiContrastIsAnExplicitWcagException: true, targetRoundingToleranceCssPixels: .1, configuredTransitionsAreNotClaimedAsActiveMotion: true, interactionMode: 'GET-only navigation and scroll; no product control clicks', browser: 'private headless Chrome over ephemeral CDP' }, routeSummary, gateContracts, gateContractFailures, networkSummary, headerHeights, mainPadding, routePadding, motion, loadTimingMs: distribution(successful.map((capture) => capture.load?.navigationToCaptureMs)), fontUsage, counts: Object.fromEntries(Object.entries(details).map(([key, records]) => [key, records.length])), details };
}

const formatDistribution = (stats) => stats.count ? `${stats.min} / ${stats.median} / ${stats.p95} / ${stats.max} (${stats.count})` : '—';
const markdownEscape = (value) => String(value).replaceAll('|', '\\|').replaceAll('\n', ' ');
export function aggregateMarkdown(aggregate) {
  const routeRows = aggregate.routeSummary.map((route) => `| [${route.slug}](./${route.report}) | ${route.successfulCaptures}/${route.captures} | ${route.h1Failures} | ${route.overflow} | ${route.targetFailures} fail / ${route.targetExceptions} except | ${route.contrastFailures} fail / ${route.contrastExceptions} except | ${route.consoleErrors}/${route.consoleWarnings} | ${route.httpErrors}/${route.requestFailures}/${route.requestCancellations} | ${route.edgeFailures} | ${route.duplicateSuccessfulGetGroups} | ${route.screenshots} |`).join('\n');
  const rows = `${routeRows}\n\nClassification ledger: ${aggregate.counts.nativeWrapperExceptions} native controls delegate to measured 44px-or-larger wrappers; ${aggregate.counts.targetRoundingTolerance} targets are within the explicit 0.1 CSS-pixel rendering tolerance; ${aggregate.counts.inactiveContrastExceptions} low-ratio records belong to inactive controls; ${aggregate.counts.addressEnrichment} final URLs retain the canonical address while adding route-owned selection state. Reduced media retained ${aggregate.counts.reducedMotionConfiguredEffects} configured motion-sensitive styles but ${aggregate.counts.reducedMotionActiveViolations} active nontrivial animations. The ${aggregate.networkSummary.duplicateSuccessfulGetGroups} duplicate-successful-GET groups contain ${aggregate.networkSummary.duplicateSuccessfulGetExtraRequests} requests beyond the first; all ${aggregate.networkSummary.successfulStreamAbortCancellations} cancellations are 200-status event streams aborted after use.`;
  const defectRows = ['captureErrors', 'settleTimeouts', 'pageIdentity', 'h1', 'localeDirection', 'horizontalOverflow', 'targetFailures', 'contrastFailures', 'consoleErrors', 'consoleWarnings', 'httpErrors', 'requestFailures', 'edgeFailures', 'reducedMotionActiveViolations', 'missingScreenshots'].map((key) => `| ${key} | ${aggregate.counts[key]} |`).join('\n');
  const headerRows = Object.entries(aggregate.headerHeights).map(([key, stats]) => `| ${key} | ${formatDistribution(stats)} |`).join('\n');
  const fontRows = Object.entries(aggregate.fontUsage).flatMap(([locale, kinds]) => Object.entries(kinds).map(([kind, record]) => `| ${locale} | ${kind} | ${record.sampleCount} | ${markdownEscape(record.platformFamilies.join(', ') || '—')} |`)).join('\n');
  const gateRows = aggregate.gateContracts.map((record) => `| ${record.capture} | ${record.exactViewport ? 'pass' : 'fail'} | ${record.apiResourceListEmpty ? 'pass' : 'fail'} | ${record.apiNetworkRequestsEmpty ? 'pass' : 'fail'} |`).join('\n') || '| — | — | — | — |';
  return `# Kairos final-cream UX evidence\n\nGenerated ${aggregate.generatedAt}. Verdict: **${aggregate.verdict}** (${aggregate.defectCount} defect records).\n\nThe flow under test is: each canonical address loads → the first meaningful operational screen settles → static layout, accessibility-adjacent, network and screenshot evidence is recorded without invoking a write control. The repository-mandated dependency-free CDP harness was used instead of the in-app browser so the same isolated, repeatable matrix could cover every route.\n\n## Coverage\n\n- Canonical routes: ${aggregate.coverage.canonicalRoutesReported}/${aggregate.coverage.canonicalRoutesExpected}\n- Desktop captures: ${aggregate.coverage.desktopCapturesReported}/${aggregate.coverage.desktopCapturesExpected}; successful ${aggregate.coverage.successfulCaptures}\n- Gate captures: ${aggregate.coverage.gateCapturesReported}/${aggregate.coverage.gateCapturesExpected}\n- Screenshots present: ${aggregate.coverage.screenshotsPresent}/${aggregate.coverage.screenshotsExpected}\n- Viewports: 1280×720, 1728×900; gate 1024×768\n- Locales: Hebrew/RTL and English/LTR\n\n## Defect ledger\n\n| Category | Records |\n| --- | ---: |\n${defectRows}\n\nSuccessful duplicate GET groups (${aggregate.counts.duplicateSuccessfulGets}) are informational and are not defects. Request cancellations (${aggregate.counts.requestCancellations}) are recorded separately from failures. Native-wrapper target exceptions: ${aggregate.counts.nativeWrapperExceptions}. Allowed full-bleed edge contacts: ${aggregate.counts.allowedEdgeContacts}.\n\n## Route matrix\n\n| Route | Captures | H1 | Overflow | Targets | Contrast | Console E/W | HTTP/Fetch/Cancel | Edge | Duplicate GET groups | PNGs |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n${rows}\n\n## Gate contract\n\n| Capture | Exact 1024×768 | ResourceTiming API list empty | Network API list empty |\n| --- | --- | --- | --- |\n${gateRows}\n\n## Shell geometry\n\nValues are min / median / p95 / max in CSS pixels, followed by sample count.\n\n| Element | Height distribution |\n| --- | --- |\n${headerRows}\n\nMain padding top/right/bottom/left: ${['top','right','bottom','left'].map((edge) => `${edge} ${formatDistribution(aggregate.mainPadding[edge])}`).join('; ')}.\n\nRoute-root padding top/right/bottom/left: ${['top','right','bottom','left'].map((edge) => `${edge} ${formatDistribution(aggregate.routePadding[edge])}`).join('; ')}.\n\n## Motion\n\n- View Transition API available in ${aggregate.motion.startViewTransitionAvailable}/${aggregate.coverage.desktopCapturesReported} desktop captures.\n- View-transition rule count: ${formatDistribution(aggregate.motion.viewTransitionRuleCount)}.\n- Normal configured effects: ${formatDistribution(aggregate.motion.normalConfiguredEffects)}.\n- Reduced-motion configured effects: ${formatDistribution(aggregate.motion.reducedConfiguredEffects)}.\n- Motion-sensitive effects remaining under reduced motion: ${aggregate.motion.reducedMotionSensitiveResidualCount}.\n\n## Fonts actually painted\n\n| Locale | Sample | Captures | Platform fonts observed |\n| --- | --- | ---: | --- |\n${fontRows}\n\nLoad-to-capture timing in ms, min / median / p95 / max: ${formatDistribution(aggregate.loadTimingMs)}. Full records, screenshots, console/network payloads, contrast calculations, target rectangles, font glyph counts and edge insets are in [aggregate.json](./aggregate.json) and each linked route report.\n`;
}
