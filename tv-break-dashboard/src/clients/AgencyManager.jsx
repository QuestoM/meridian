import React from 'react';
import ClientsWorkspace from './ClientsWorkspace';

// The Agencies navigation entry now opens the Clients destination on the agency
// records view, which is the same panel this file used to be: its body moved to
// AgencyRecordsPanel.jsx unchanged, line for line.
//
// The entry keeps its name, its hash and its first screen, so nothing that was
// zero clicks from here is further away now.

export default function AgencyManager(props) {
  return <ClientsWorkspace view="agencies" {...props} />;
}
