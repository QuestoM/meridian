import React from 'react';
import { Button } from '@mui/material';
import { pageText } from '../shell/format';
import { NO_CHANNEL, shortReason } from './today-scope';

// The refusal a block renders where it would otherwise print a figure it cannot
// report as the operator's. Three beats, in the order a reader needs them: what
// is missing here, why, and the one control that ends it. It is one component
// so that every block on this screen refuses in the same words, and so that a
// block added later cannot invent a softer sentence.
//
// The lead is the block's own, because "these four figures" and "the money
// story" are different absences and a reader has to know which one is in front
// of them. The cause and the control are the screen's, never the block's.
export function ChannelRefusal({ locale, lead, onOpenSettings }) {
  const needs = pageText(locale, NO_CHANNEL.needs_en, NO_CHANNEL.needs_he);
  return (
    <div className="today-unattributed">
      <p className="today-unattributed-lead">{lead}</p>
      <p className="today-unattributed-cause">{shortReason(locale)}</p>
      {onOpenSettings ? (
        <Button className="today-secondary" type="button" onClick={onOpenSettings}>
          {needs}
        </Button>
      ) : (
        // No handler is still a path forward: the sentence names where the
        // channel is set, so the block never ends on the cause alone.
        <p className="today-unattributed-needs">{needs}</p>
      )}
    </div>
  );
}

export default ChannelRefusal;
