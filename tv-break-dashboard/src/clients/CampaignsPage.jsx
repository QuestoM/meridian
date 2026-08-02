import React from 'react';
import ClientsWorkspace from './ClientsWorkspace';

// The Campaigns navigation entry now opens the Clients destination on the
// campaigns view, which carries two things rather than one: the campaigns
// booked in this product, with their flights, and the historical rollup this
// page used to be, which moved to CampaignRollupPanel.jsx with its honest empty
// state intact.

export function CampaignsPage(props) {
  return <ClientsWorkspace view="campaigns" {...props} />;
}

export default CampaignsPage;
