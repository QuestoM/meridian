# Kairos UX evidence harness

Run this against the read-only local runtime (`KAIROS_AUTH_DISABLED=1`,
`KAIROS_PLAN_READONLY=1`, and all provider credentials empty). The harness uses
only GET navigation and scrolling; it never clicks a product control.

```sh
node scripts/capture-ux-evidence.mjs \
  --base-url http://127.0.0.1:3000/ \
  --out /tmp/kairos-ux-evidence
```

The default run captures all 34 canonical addresses in Hebrew/RTL and
English/LTR at 1280×720 and 1728×900. `--routes today,plan-objective` selects a
stable subset. `--gate` alone records the true 1024×768 desktop-required gate;
combine it with `--routes …` to add the gate to a route subset, or use
`--all --gate` for the complete acceptance suite. Use
`--list-routes` to print every stable slug.

After a complete run, `--resume --routes …` replaces only those route reports
and rebuilds the aggregate from every canonical report already in the output
directory. Add `--gate` to replace the gate in the same corrective run.

Each route gets `report.json` plus full-page, top, middle, and bottom PNGs. The
report also measures shell/header/navigation heights, main and route padding,
logical four-edge insets, under-12px edge contacts (with a named full-bleed
timeline/table allowlist), and normal versus reduced-motion computed styles.
The root `index.json` records the run contract. Locale simulation is isolated to
the browser profile: before application code runs, the harness sets the local
preference and wraps `fetch` so that only `locale` and `direction` fields in
JSON from `/api/overview` and `/api/settings` change; status, headers, and every
other field remain intact. No locale-setting request is sent to the backend.

Chrome/Chromium is auto-discovered. Set `CHROME_PATH` when it lives elsewhere.
Every run owns a temporary profile and ephemeral CDP port and removes both on
exit.
