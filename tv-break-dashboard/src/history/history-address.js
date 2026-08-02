// The address of one entry, what the list has to be showing for that address to
// be reachable, and why it is not when it is not.
//
// A page is a window on the record rather than the record: the endpoint caps a
// page at 500 entries. Measured on the running instance, the record holds 5,323
// entries and the newest 500 of them span sixteen minutes, so an entry recorded
// this morning is already off the unfiltered page. That is the fact every
// decision here turns on.
//
// This lives apart from the surface so it can be executed by a test rather than
// read by one. Four different things can be true when an asked-for entry is not
// on screen, two of them have a control that fixes them and two do not, and a
// note that names the wrong one is worse than no note.

import { isoDay } from './history-labels.js';

export const DEFAULT_LIMIT = 200;
export const WIDE_LIMIT = 500;

// A restore point is the one entry that other records link to by name: a
// restore names the point it came from and the point that undoes it.
export const POINT_PREFIX = 'version:';

export function isPointAddress(address) {
  return String(address || '').startsWith(POINT_PREFIX);
}

export function pointAddress(versionId) {
  return versionId ? `${POINT_PREFIX}${versionId}` : '';
}

// Where the list has to be standing for an address to be reachable.
//
// A point address opens on the restore points, because that is the only list
// that can promise to hold it: there are 200 points and a page holds 500, while
// an unfiltered page holds the newest 500 entries of every kind and drops points
// that are minutes old. Measured: from the Restore filter, and cold from a
// shared link, the point version:1337540bd866 is absent from an unfiltered page
// and present on the points. Every other address opens on everything.
export function addressQuery(address) {
  return {
    kind: isPointAddress(address) ? 'restore_point' : '',
    actor: '',
    needle: '',
    limit: WIDE_LIMIT,
  };
}

// Why the asked-for entry is not on screen. One of four, in the order that puts
// the most specific true statement first.
//
// point_gone: the list is the whole set of points and the point is not in it,
//   so no control on this surface can find it and the note offers none.
// filtered: a filter the reader can drop is hiding it.
// paged_out: it is real and older than the page.
// absent: nothing is filtered, nothing is paged out, and it is not in the record.
// The day an address is on, when the address carries one. A change, a restore
// and an account event are addressed by their own timestamp, so a link into a
// part of the record this page does not reach can be answered with the day it
// is on rather than with a refusal. A restore point and a run are addressed by
// an opaque id and carry no day, so this says so by returning nothing and the
// note offers the reader the controls instead of a jump that would guess.
//
// The stamp inside an id is UTC and the day control is a broadcast day, so the
// conversion is the surface's own, shared with every other date on this screen.
const STAMP_IN_ADDRESS = /(\d{4}-\d{2}-\d{2}T[\d:.]+(?:[+-]\d{2}:\d{2}|Z)?)/;

export function dayOfAddress(address) {
  const found = STAMP_IN_ADDRESS.exec(String(address || ''));
  return found ? isoDay(found[1]) : '';
}

export function missedReason({ wanted, kind, actor, needle, pagedOut }) {
  const text = String(needle || '').trim();
  const filtered = Boolean(kind || actor || text);
  if (isPointAddress(wanted) && kind === 'restore_point' && !actor && !text && !pagedOut) return 'point_gone';
  if (filtered) return 'filtered';
  if (pagedOut) return 'paged_out';
  return 'absent';
}
