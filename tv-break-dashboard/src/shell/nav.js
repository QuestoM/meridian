import {
  Activity,
  Bot,
  Building2,
  CalendarClock,
  CalendarDays,
  ClipboardCheck,
  Coins,
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
  ['Calendar', CalendarClock],
  ['Reports', ListChecks],
  ['Data', Database],
  ['Advertisers', Users],
  ['Agencies', Building2],
  ['Pricing', Coins],
  ['Overrides', SlidersHorizontal],
  ['Assistant', Bot],
  ['Versions', History],
  ['Settings', Settings],
];

export function viewFromLocation() {
  if (typeof window === 'undefined') {
    return 'Overview';
  }
  const hash = decodeURIComponent(window.location.hash.replace(/^#/, ''));
  return navItems.some(([label]) => label === hash) ? hash : 'Overview';
}

export function gridAxisFromLocation() {
  if (typeof window === 'undefined') {
    return 'day';
  }
  const axis = new URLSearchParams(window.location.search).get('axis');
  return ['day', 'daypart', 'hour', 'type'].includes(axis) ? axis : 'day';
}
