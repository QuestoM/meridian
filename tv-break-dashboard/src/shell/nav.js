import {
  Activity,
  Bot,
  Building2,
  CalendarDays,
  ClipboardCheck,
  Database,
  FileBarChart,
  Gauge,
  History,
  LayoutGrid,
  ListChecks,
  Settings,
  SlidersHorizontal,
  TableProperties,
  Users,
} from 'lucide-react';

export const navItems = [
  ['Overview', LayoutGrid],
  ['Optimizer', Activity],
  ['Schedule', CalendarDays],
  ['Inventory', TableProperties],
  ['Break Library', ClipboardCheck],
  ['Campaigns', FileBarChart],
  ['Forecasts', Gauge],
  ['Reports', ListChecks],
  ['Data', Database],
  ['Advertisers', Users],
  ['Agencies', Building2],
  ['Overrides', SlidersHorizontal],
  ['Assistant', Bot],
  ['Versions', History],
  ['Settings', Settings],
];

// Routes that were removed from the rail because they now live inside a
// destination as a tab. These labels are still valid for viewFromLocation so an
// existing bookmark still reaches something sensible (the router redirects them).
export const removedRoutes = ['Calendar', 'Pricing'];

// All labels that a hash may legitimately name, including removed rail entries
// whose bookmarks are still honoured by the workspace router's redirect logic.
const allKnownLabels = new Set([
  ...navItems.map(([label]) => label),
  ...removedRoutes,
]);

export function viewFromLocation() {
  if (typeof window === 'undefined') {
    return 'Overview';
  }
  const hash = decodeURIComponent(window.location.hash.replace(/^#/, ''));
  return allKnownLabels.has(hash) ? hash : 'Overview';
}

export function gridAxisFromLocation() {
  if (typeof window === 'undefined') {
    return 'day';
  }
  const axis = new URLSearchParams(window.location.search).get('axis');
  return ['day', 'daypart', 'hour', 'type'].includes(axis) ? axis : 'day';
}
