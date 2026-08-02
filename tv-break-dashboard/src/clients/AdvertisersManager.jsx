import React from 'react';
import ClientsWorkspace from './ClientsWorkspace';

// The Advertisers navigation entry now opens the Clients destination on the
// advertiser records view, which is the same panel this file used to be: its
// body moved to AdvertiserRecordsPanel.jsx unchanged, line for line, and the
// four other views of the destination are one click away from it.
//
// The entry keeps its name, its hash and its first screen, so nothing that was
// zero clicks from here is further away now.

export default function AdvertisersManager(props) {
  return <ClientsWorkspace view="advertisers" {...props} />;
}
