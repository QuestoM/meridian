import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';

// Page awareness for the assistant dock. The dashboard registers the active
// view through the provider's page prop (top-down), and pages report their
// focused entity through useAssistantEntity (bottom-up). The assistant panel
// reads both and sends them as the advisory page_context on every ask, per the
// frozen contract: { view, label, entity: { type, id, label } | null }.
// Everything degrades to null when no provider is mounted, so components stay
// usable in isolation and an absent context sends no page_context at all.
// Plain .js on purpose (no JSX): manager pages import the hooks from here and
// the file must stay loadable regardless of the JSX transform.

const PageStateContext = createContext(null);
const PageActionsContext = createContext(null);

export function AssistantPageProvider({ page, children }) {
  const [entity, setEntity] = useState(null);
  // The actions object is stable for the provider's lifetime, so the entity
  // hook's effect never re-fires because of a dashboard re-render.
  const actions = useMemo(() => ({ setEntity }), []);
  const view = page && page.view ? String(page.view) : '';
  const label = page && page.label ? String(page.label) : '';
  // Keyed on the primitive fields, not the page object identity, so consumers
  // do not re-render just because the dashboard rebuilt an equal props object.
  const value = useMemo(() => ({ page: view ? { view, label } : null, entity }), [view, label, entity]);
  return React.createElement(
    PageActionsContext.Provider,
    { value: actions },
    React.createElement(PageStateContext.Provider, { value }, children),
  );
}

// Read the current location: { page: { view, label }, entity } or null.
export function useAssistantPage() {
  return useContext(PageStateContext);
}

// Read just the page: { view, label } or null. Managers can use this to know
// where they are without also subscribing to the entity registration.
export function useAssistantPageView() {
  const state = useContext(PageStateContext);
  return state && state.page ? state.page : null;
}

// A page with an open record calls this with the record's type, store id and
// display name. Registration happens on mount and on every change of the
// arguments; the record is cleared on unmount or when the id empties, and a
// stale clear never wipes a newer registration from another surface.
export function useAssistantEntity(type, id, label) {
  const actions = useContext(PageActionsContext);
  useEffect(() => {
    if (!actions) return undefined;
    if (!type || id === null || id === undefined || id === '') return undefined;
    const record = { type: String(type), id: String(id), label: label ? String(label) : '' };
    actions.setEntity(record);
    return () => {
      actions.setEntity((current) => (current && current.type === record.type && current.id === record.id ? null : current));
    };
  }, [actions, type, id, label]);
}

// The frozen-contract body fragment for an ask. Null when there is no page to
// report, so the request degrades to exactly today's behavior.
export function buildPageContext(state) {
  if (!state || !state.page || !state.page.view) return null;
  return {
    view: String(state.page.view),
    label: state.page.label ? String(state.page.label) : '',
    entity: state.entity ? { type: state.entity.type, id: state.entity.id, label: state.entity.label } : null,
  };
}
