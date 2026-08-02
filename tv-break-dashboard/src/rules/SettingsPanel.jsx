import React from 'react';
import RulesWorkspace from './RulesWorkspace';

// The Rules destination's entry point. This module used to be a settings page
// carrying eight unrelated panels in one scroll, with the programming
// representative's constraint builder at character 4,240 of it, below the
// optimizer's risk lever and a pacing denominator floor.
//
// It now opens the Rules workspace, which holds the same controls sorted by the
// job that uses them: restrictions, the licence, the rate card, the channel and
// model declarations, and the planning levers. Nothing that could be set here
// before is unreachable, and the levers section is the old page unchanged.

export function SettingsPanel(props) {
  return <RulesWorkspace {...props} />;
}

export default SettingsPanel;
