import React from 'react';

const GLYPHS = {
  today: (
    <>
      <circle cx="12" cy="12" r="8.25" />
      <path d="M4.4 13.5h15.2M7 13.5c1.5-3.8 3.2-5.7 5-5.7s3.5 1.9 5 5.7" />
      <circle cx="12" cy="9.2" r="1" data-fill="true" />
    </>
  ),
  plan: (
    <>
      <path d="M4 5.5h16v13H4zM8 3.5v4M16 3.5v4M4 9.5h16" />
      <path d="M7 13h3M12 13h5M7 16h6" />
    </>
  ),
  broadcast: (
    <>
      <path d="M12 18.5v-9M8.5 20h7M10 9.5a2 2 0 1 1 4 0 2 2 0 0 1-4 0Z" />
      <path d="M7.7 5.2a6.1 6.1 0 0 0 0 8.6M16.3 5.2a6.1 6.1 0 0 1 0 8.6M5 2.6a9.8 9.8 0 0 0 0 13.8M19 2.6a9.8 9.8 0 0 1 0 13.8" />
    </>
  ),
  commercial: (
    <>
      <path d="M4 19.5h16M5.5 17V9.5h3V17M10.5 17V5h3v12M15.5 17v-4.5h3V17" />
      <path d="M4.5 7.2 9 4.5l4 1.2 6.5-3.2" />
    </>
  ),
  sources: (
    <>
      <ellipse cx="12" cy="5.5" rx="7.5" ry="3" />
      <path d="M4.5 5.5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6M4.5 11.5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6" />
      <path d="M8.5 9.8h7" />
    </>
  ),
  governance: (
    <>
      <path d="M12 3.5 19 6v5.2c0 4.4-2.6 7.6-7 9.3-4.4-1.7-7-4.9-7-9.3V6l7-2.5Z" />
      <path d="m8.6 12.1 2.2 2.2 4.8-5" />
    </>
  ),
  history: (
    <>
      <circle cx="12" cy="12" r="8.2" />
      <path d="M12 7.2v5.2l3.6 2.1M5.8 5.8 3.7 4.6M3.8 4.5v3" />
    </>
  ),
  mabat: (
    <>
      <path d="M3.4 12c2.5-3.8 5.4-5.7 8.6-5.7s6.1 1.9 8.6 5.7c-2.5 3.8-5.4 5.7-8.6 5.7S5.9 15.8 3.4 12Z" />
      <path d="M12 8.7a3.3 3.3 0 1 0 3.3 3.3" />
      <path d="M12 8.7v6.6" />
    </>
  ),
};

export function KairosIcon({ name, size = 20, className = '', title, ...rest }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`kairos-icon ${className}`.trim()}
      aria-hidden={title ? undefined : 'true'}
      role={title ? 'img' : undefined}
      {...rest}
    >
      {title ? <title>{title}</title> : null}
      {GLYPHS[name] || GLYPHS.mabat}
    </svg>
  );
}

export function KairosMark({ size = 36, className = '', title = 'Kairos' }) {
  return (
    <svg
      viewBox="0 0 32 32"
      width={size}
      height={size}
      fill="currentColor"
      className={`kairos-mark ${className}`.trim()}
      aria-hidden={title ? undefined : 'true'}
      role={title ? 'img' : undefined}
      focusable="false"
    >
      {title ? <title>{title}</title> : null}
      <path d="M4 3h11v8.5l3.5 3.5-3.5 3.5V29H4V3Z" />
      <path d="M17 3h11v26H17V18.5l3.5-3.5-3.5-3.5V3Z" />
    </svg>
  );
}

function iconComponent(name) {
  return function DomainIcon(props) {
    return <KairosIcon name={name} {...props} />;
  };
}

export const TodayIcon = iconComponent('today');
export const PlanIcon = iconComponent('plan');
export const BroadcastIcon = iconComponent('broadcast');
export const CommercialIcon = iconComponent('commercial');
export const SourcesIcon = iconComponent('sources');
export const GovernanceIcon = iconComponent('governance');
export const HistoryIcon = iconComponent('history');
export const MabatIcon = iconComponent('mabat');
export const KaiIcon = MabatIcon;
