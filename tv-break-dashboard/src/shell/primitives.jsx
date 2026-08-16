import React from 'react';
import { Tooltip } from '@mui/material';
import {
  ArrowDown,
  ArrowUp,
  CircleAlert,
  Inbox,
  LoaderCircle,
} from 'lucide-react';
import { Numeric, pageText } from './format';
import { cx } from './dom-controls';
export { DataTable, LazyDataGrid } from './data-table';
export { Dialog, Sheet, focusFirstWithin, useFocusReturn } from './modal-primitives';
export { InputControl, Pressable, SelectControl, TextAreaControl, cx } from './dom-controls';

export function Toast({ className = '', role = 'status', live = 'polite', children, ...rest }) {
  return (
    <div className={cx('card', 'studio-toast', className)} role={role} aria-live={live} aria-atomic="true" {...rest}>
      {children}
    </div>
  );
}

const STATUS_TONES = {
  ready: 'positive',
  compliant: 'positive',
  live: 'positive',
  success: 'positive',
  positive: 'positive',
  at_risk: 'warning',
  attention: 'warning',
  wired_off: 'warning',
  warning: 'warning',
  error: 'danger',
  danger: 'danger',
  absent: 'neutral',
  empty: 'neutral',
  neutral: 'neutral',
  info: 'info',
};

function statusTone(status, explicitTone) {
  if (explicitTone) return STATUS_TONES[explicitTone] || explicitTone;
  return STATUS_TONES[String(status || 'neutral').toLowerCase()] || 'neutral';
}

export function Status({
  status = 'neutral',
  tone,
  icon,
  className = '',
  children,
  ...rest
}) {
  const resolvedTone = statusTone(status, tone);
  return (
    <span className={cx('studio-status', `studio-status--${resolvedTone}`, className)} data-status={status} {...rest}>
      <span className="studio-status__mark" data-icon={icon ? 'true' : undefined} aria-hidden="true">{icon || null}</span>
      <span>{children}</span>
    </span>
  );
}

export function Metric({ label, value, delta, sub, icon: Icon, positive = false, tone, title }) {
  const hasDelta = delta !== undefined && delta !== null && delta !== '';
  const body = (
    <div className="metric studio-metric">
      {Icon ? (
        <span className={cx('metric-icon', tone)} aria-hidden="true">
          <Icon size={20} strokeWidth={1.75} />
        </span>
      ) : null}
      <span className="metric-copy">
        <span>{label}</span>
        <strong><Numeric>{value}</Numeric></strong>
        {sub ? <small className="metric-sub">{sub}</small> : null}
      </span>
      {hasDelta ? (
        <span className={positive ? 'delta positive' : tone === 'risk' ? 'delta risk' : 'delta negative'}>
          {positive ? <ArrowUp size={12} aria-hidden="true" /> : tone === 'risk' ? null : <ArrowDown size={12} aria-hidden="true" />}
          <Numeric>{delta}</Numeric>
        </span>
      ) : null}
    </div>
  );
  return title ? <Tooltip title={title} placement="bottom">{body}</Tooltip> : body;
}

/* Canonical bounded material surface. */
export function Card({ as: Tag = 'section', dense = false, className = '', children, ...rest }) {
  return <Tag className={cx('card', dense && 'card-dense', className)} {...rest}>{children}</Tag>;
}

export function CardBody({ children, className = '' }) {
  return <div className={cx('card-body', className)}>{children}</div>;
}

export function CardBleed({ children, className = '' }) {
  return <div className={cx('card-bleed', className)}>{children}</div>;
}

export function PageHeader({ locale, titleEn, titleHe, bodyEn, bodyHe, action }) {
  return (
    <header className="page-header studio-context-header">
      <div>
        <h1>{pageText(locale, titleEn, titleHe)}</h1>
        <p>{pageText(locale, bodyEn, bodyHe)}</p>
      </div>
      {action ? <div className="studio-context-header__action">{action}</div> : null}
    </header>
  );
}

export function StatusBadge({ status, locale, mode = 'inline' }) {
  const normalized = String(status || 'ready').toLowerCase().replace(/[^a-z0-9_-]/g, '-');
  const labelMap = {
    ready: pageText(locale, 'Ready', 'מוכן'),
    compliant: pageText(locale, 'Compliant', 'תקין'),
    at_risk: pageText(locale, 'Needs review', 'דורש בדיקה'),
    attention: pageText(locale, 'Needs attention', 'דורש טיפול'),
    empty: pageText(locale, 'No rows yet', 'אין שורות עדיין'),
    error: pageText(locale, 'Error', 'שגיאה'),
    live: pageText(locale, 'Live', 'פעיל'),
    wired_off: pageText(locale, 'Wired, off', 'מחווט-כבוי'),
    absent: pageText(locale, 'Not built', 'לא קיים'),
  };
  return (
    <Status
      status={normalized}
      className={cx('status-badge', mode, normalized)}
    >
      {labelMap[normalized] || status}
    </Status>
  );
}

function StateLayout({ kind, icon, title, description, action, role, live = 'polite', className = '', children }) {
  return (
    <div
      className={cx('studio-state', `studio-state--${kind}`, className)}
      role={role}
      aria-live={role ? live : undefined}
    >
      <span className="studio-state__icon" aria-hidden="true">{icon}</span>
      <div className="studio-state__copy">
        <strong>{title}</strong>
        {description ? <p>{description}</p> : null}
        {children}
      </div>
      {action ? <div className="studio-state__action">{action}</div> : null}
    </div>
  );
}

export function EmptyState({ title, description, action, className = '', children }) {
  return (
    <StateLayout
      kind="empty"
      icon={<Inbox size={24} strokeWidth={1.75} />}
      title={title}
      description={description}
      action={action}
      className={className}
    >
      {children}
    </StateLayout>
  );
}

export function ErrorState({ title, description, action, className = '', children }) {
  return (
    <StateLayout
      kind="error"
      icon={<CircleAlert size={24} strokeWidth={1.75} />}
      title={title}
      description={description}
      action={action}
      role="alert"
      live="assertive"
      className={className}
    >
      {children}
    </StateLayout>
  );
}

export function LoadingState({ title, description, className = '', children }) {
  return (
    <StateLayout
      kind="loading"
      icon={<LoaderCircle className="studio-state__spinner" size={24} strokeWidth={1.75} />}
      title={title}
      description={description}
      role="status"
      className={className}
    >
      {children}
    </StateLayout>
  );
}
