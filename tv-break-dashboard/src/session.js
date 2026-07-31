// Who am I, what is my affiliation, and what may I do.
//
// Every surface asks these three questions and today none of them can: the
// dashboard fetches the account record and reads only its name and role, so a
// walled control renders enabled and fails after the click. This module is the
// single answer, and it is deliberately dependency-free so it can be imported
// from any tree without pulling a component with it.
//
// The permission rule it encodes is the product's, not this file's.
// Affiliation decides which side of the line an account can see. Role decides
// what it can change on its side. And a walled control always prefers the
// can_edit its own endpoint sent over anything derived here, because the
// server is the authority and this is the fallback for an endpoint that has
// not been given a can_edit yet.
//
// The job field is the third question's other half. Role says what an account
// may write; job says what this person's work is, which decides where they
// land and nothing else. A wrong job costs somebody a good first screen, never
// their access.

export const API_BASE = import.meta.env.VITE_KAIROS_API_URL || '';

export const UNSET_JOB = 'unset';

export const WRITE_ROLES = ['admin', 'operator'];

// The thirteen doors. A door is a named landing view, not a destination: five
// of them are views of one workspace. The map is complete here because this
// file freezes at the end of wave zero; a piece implements the route for its
// own door id and adds nothing to this list.
export const DOORS = {
  today: {
    workspace: 'today',
    en: 'Today',
    he: 'היום',
  },
  'plan.week': {
    workspace: 'plan',
    en: 'Plan, week',
    he: 'תוכנית, שבוע',
  },
  'plan.day': {
    workspace: 'plan',
    en: 'Plan, day',
    he: 'תוכנית, יום',
  },
  'plan.tonight': {
    workspace: 'plan',
    en: "Plan, tonight's breaks",
    he: 'תוכנית, הברייקים של הערב',
  },
  'rules.restrictions': {
    workspace: 'rules',
    en: 'Rules, restrictions',
    he: 'כללים, הגבלות',
  },
  'rules.licence': {
    workspace: 'rules',
    en: 'Rules, the licence',
    he: 'כללים, הרישיון',
  },
  'rules.rate_card': {
    workspace: 'rules',
    en: 'Rules, the rate card',
    he: 'כללים, המחירון',
  },
  'clients.all': {
    workspace: 'clients',
    en: 'Clients, all clients',
    he: 'לקוחות, כל הלקוחות',
  },
  'clients.campaigns': {
    workspace: 'clients',
    en: 'Clients, campaigns on air',
    he: 'לקוחות, קמפיינים באוויר',
  },
  'clients.money': {
    workspace: 'clients',
    en: 'Clients, delivered money',
    he: 'לקוחות, הכנסה שסופקה',
  },
  'sources.today': {
    workspace: 'sources',
    en: "Sources, today's inputs",
    he: 'מקורות, הקלטים של היום',
  },
  'account.accounts': {
    workspace: 'account',
    en: 'Account menu, accounts',
    he: 'תפריט החשבון, חשבונות',
  },
  'model.console': {
    workspace: 'model',
    en: 'Model console',
    he: 'קונסולת המודל',
  },
};

// The thirteen jobs, in the order the picker lists them. companyOnly hides the
// model steward's row from a channel account, so nothing tells that account
// the other side of the line exists.
export const JOBS = [
  {
    id: 'general_manager',
    door: 'today',
    en: 'General manager',
    he: 'מנכ"ל',
  },
  {
    id: 'planner',
    door: 'plan.week',
    en: 'Planner',
    he: 'מתכנן לוח',
  },
  {
    id: 'scheduler',
    door: 'plan.day',
    en: 'Scheduler',
    he: 'משבץ ברייקים',
  },
  {
    id: 'traffic_operator',
    door: 'plan.tonight',
    en: 'Traffic operator',
    he: 'טראפיק',
  },
  {
    id: 'programming_representative',
    door: 'rules.restrictions',
    en: 'Programming representative',
    he: 'נציג מחלקת תוכן',
  },
  {
    id: 'compliance_owner',
    door: 'rules.licence',
    en: 'Compliance owner',
    he: 'אחראי רגולציה',
  },
  {
    id: 'yield_owner',
    door: 'rules.rate_card',
    en: 'Revenue and yield owner',
    he: 'אחראי הכנסות ותשואה',
  },
  {
    id: 'account_manager',
    door: 'clients.all',
    en: 'Account manager',
    he: 'מנהל לקוח',
  },
  {
    id: 'campaign_manager',
    door: 'clients.campaigns',
    en: 'Campaign manager',
    he: 'מנהל קמפיינים',
  },
  {
    id: 'analyst',
    door: 'clients.money',
    en: 'Analyst',
    he: 'אנליסט',
  },
  {
    id: 'data_steward',
    door: 'sources.today',
    en: 'Data steward',
    he: 'אחראי נתונים',
  },
  {
    id: 'account_administrator',
    door: 'account.accounts',
    en: 'Account administrator',
    he: 'מנהל חשבונות',
  },
  {
    id: 'model_steward',
    door: 'model.console',
    en: 'Model steward',
    he: 'אחראי המודל',
    companyOnly: true,
  },
];

export const JOB_IDS = JOBS.map((job) => job.id);

// The refusals, in the words the server sends. A control renders the reason
// before the click; the server sends the same string if the click happens
// anyway. A test pins these against the Python constants so they cannot drift.
export const WALLS = {
  events: {
    companyOnly: true,
    detail: 'עריכת אירועים שמורה לצוות החברה',
  },
  eventPricing: {
    companyOnly: true,
    detail: 'הפעלת תמחור אירועים שמורה לצוות החברה',
  },
  audienceActivation: {
    companyOnly: true,
    detail: 'הפעלת מודל הקהל שמורה לצוות החברה',
  },
  guardrails: {
    companyOnly: false,
    adminOnly: true,
    detail: 'עריכת מגבלות הרגולציה שמורה למנהל המערכת',
  },
  companySurface: {
    companyOnly: true,
    detail: 'התצוגה הזו שמורה לצוות החברה',
  },
  readOnlyRole: {
    companyOnly: false,
    detail: 'לחשבון צפייה אין הרשאת עריכה',
  },
};

// An account record from GET /api/auth/me, normalized. Auth off is an honest
// state of its own, not a fake identity: the shell says login is not set up
// and every gate reads open, exactly as the server behaves.
export function normalizeSession(body) {
  const record = body && typeof body === 'object' ? body : {};
  const authDisabled = record.auth_disabled === true;
  const role = String(record.role || '');
  const affiliation = record.affiliation === 'channel' ? 'channel' : 'company';
  const job = String(record.job || UNSET_JOB) || UNSET_JOB;
  return {
    authDisabled,
    username: String(record.username || ''),
    displayName: String(record.display_name || record.username || ''),
    role,
    affiliation,
    job: JOB_IDS.includes(job) ? job : UNSET_JOB,
    mustChangePassword: record.must_change_password === true,
    isCompany: authDisabled || affiliation === 'company',
    isAdmin: authDisabled || role === 'admin',
    canWrite: authDisabled || WRITE_ROLES.includes(role),
  };
}

export const ANONYMOUS_SESSION = normalizeSession({ auth_disabled: true });

export async function fetchSession() {
  try {
    const response = await fetch(`${API_BASE}/api/auth/me`, { credentials: 'include' });
    if (!response.ok) {
      return { ok: false, status: response.status, session: null };
    }
    return { ok: true, status: response.status, session: normalizeSession(await response.json()) };
  } catch {
    return { ok: false, status: 0, session: null };
  }
}

// The gate as this session sees it, for a wall declared in WALLS. Returns the
// same shape the server stamps, so a caller reads one contract either way.
export function sessionCanEdit(session, wall) {
  const account = session || ANONYMOUS_SESSION;
  const rule = wall || {};
  if (rule.companyOnly && !account.isCompany) {
    return { canEdit: false, reason: rule.detail || WALLS.companySurface.detail };
  }
  if (rule.adminOnly && !account.isAdmin) {
    return { canEdit: false, reason: rule.detail || WALLS.readOnlyRole.detail };
  }
  if (!rule.adminOnly && !account.canWrite) {
    return { canEdit: false, reason: WALLS.readOnlyRole.detail };
  }
  return { canEdit: true, reason: null };
}

// The authority is the endpoint. A payload that carries can_edit decides, and
// its reason is used verbatim; a payload without one falls back to the session
// rule, which is what an endpoint that has not been upgraded yet needs.
export function payloadCanEdit(payload, session, wall) {
  const body = payload && typeof payload === 'object' ? payload : {};
  if (typeof body.can_edit === 'boolean') {
    return {
      canEdit: body.can_edit,
      reason: body.can_edit ? null : String(body.can_edit_reason || (wall && wall.detail) || ''),
    };
  }
  return sessionCanEdit(session, wall);
}

export function jobFor(session) {
  const account = session || ANONYMOUS_SESSION;
  return JOB_IDS.includes(account.job) ? account.job : UNSET_JOB;
}

// The door this account lands on. An unset job lands on Today, where the
// picker is, so a new starter is never dropped on somebody else's screen.
export function doorFor(session) {
  const job = JOBS.find((entry) => entry.id === jobFor(session));
  if (!job) return 'today';
  if (job.companyOnly && !(session || ANONYMOUS_SESSION).isCompany) return 'today';
  return job.door;
}

export function doorLabel(doorId, locale = 'he') {
  const door = DOORS[doorId];
  if (!door) return '';
  return locale === 'en' ? door.en : door.he;
}

export function jobLabel(jobId, locale = 'he') {
  const job = JOBS.find((entry) => entry.id === jobId);
  if (!job) return '';
  return locale === 'en' ? job.en : job.he;
}

// With login not set up there is no account record to write a job to, so the
// picker does not render and nobody is asked a question the server cannot keep.
export function needsJobPicker(session) {
  const account = session || ANONYMOUS_SESSION;
  return !account.authDisabled && jobFor(account) === UNSET_JOB;
}

// The picker card's rows: the job in the person's own word, and the door it
// opens. A channel account never sees the company-only row.
export function jobPickerRows(session, locale = 'he') {
  const account = session || ANONYMOUS_SESSION;
  return JOBS.filter((job) => !job.companyOnly || account.isCompany).map((job) => ({
    id: job.id,
    door: job.door,
    label: locale === 'en' ? job.en : job.he,
    doorLabel: doorLabel(job.door, locale),
  }));
}

// Self-service by design: a job decides a landing screen, not a permission, so
// the person choosing it needs no administrator and a viewer may set their own.
export async function saveJob(job) {
  try {
    const response = await fetch(`${API_BASE}/api/auth/job`, {
      method: 'PUT',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job }),
    });
    const data = response.ok ? await response.json() : null;
    return { ok: response.ok, status: response.status, session: data ? normalizeSession(data) : null };
  } catch {
    return { ok: false, status: 0, session: null };
  }
}
