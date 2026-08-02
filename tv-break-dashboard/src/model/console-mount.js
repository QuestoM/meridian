// The one side-effect import that puts the model console on the page.
//
// It is imported from both files in this tree that the application already
// evaluates at boot, so the console survives either of the two chains being
// removed by the piece that owns the other end. Nothing here runs before the
// document exists and nothing renders until the session answers company.
//
// When the shell carries the context switcher, this file and the two imports of
// it go away together. It is deliberately a separate module so that removal is
// a deletion rather than an edit inside a component.

import { mountModelConsole } from './console-bridge.jsx';

mountModelConsole();
