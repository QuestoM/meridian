import React from 'react';
import { Button } from '../../studio/actions';
import { Plus, ShieldCheck, Upload } from 'lucide-react';
import { amount, isolate, pick } from './pacing-helpers';

// The two controls a board row offers, split out of PacingRow.jsx on the
// component boundary when that file reached the size law. They are the acts; the
// row keeps the readings.
//
// The destination a traffic file is uploaded at, in the shell's own address form
// and under the shell's own name for it. This is a link and not a callback: the
// shell reads the hash and this piece owns no seam into its router, so the
// address contract is the thing both sides already agree on.
const UPLOAD_HASH = '#Data';

// Which missing thing is fixed where. A row whose pace could not be stated names
// what is missing and the path forward, and until now it named no control at
// all: the five rows reading "Open the campaign and set a goal on its flight"
// offered nothing to open it with, while every at-risk row got a named remedy.
// A diagnosis with a path and no door is the defect this row exists not to
// repeat, so the two codes that point at a campaign and the three that point at
// an upload each carry the control that performs them.
//
// `unmeasurable` deliberately carries none. Its path forward is to supply a
// panel breakdown for an audience, and no screen in this product does that, so
// naming a door that does not exist would be worse than naming none.
const OPENS_THE_CAMPAIGN = ['no_goal', 'no_flight_dates'];
const OPENS_THE_UPLOAD = ['no_source', 'gap_in_elapsed', 'not_started'];

export function Remedy({ remedy, locale, busy, onRaise, onOpenMakeGood, onOpenCampaign }) {
  if (remedy.kind === 'raise') {
    const value = amount(remedy.value, remedy.unit, locale);
    return (
      <Button type="button" className="pacing-remedy" disabled={busy} onClick={onRaise}>
        <Plus size={13} aria-hidden="true" />
        {pick(locale, `Raise a make-good for ${value}`, `פתחו פיצוי שידור על ${value}`)}
      </Button>
    );
  }
  if (remedy.kind === 'open') {
    return (
      <Button type="button" className="pacing-remedy" onClick={() => onOpenMakeGood(remedy.makeGoodId)}>
        {pick(locale, `Open make-good ${remedy.makeGoodId}`, `פתחו את פיצוי ${isolate(remedy.makeGoodId)}`)}
      </Button>
    );
  }
  // The statement carries the act. A remedy that names an upload and then leaves
  // the reader to find the upload themselves is a diagnosis, not a remedy, so the
  // act this row offers is the one control that performs it.
  if (remedy.kind === 'book') {
    return (
      <a className="pacing-remedy" href={UPLOAD_HASH}>
        <Upload size={13} aria-hidden="true" />
        {/* The Hebrew names the screen it opens. את before a bare noun is not
            Hebrew: the destination is נתונים and the phrase has to say so. */}
        {pick(locale, 'Open Data to upload it', 'פתחו את מסך הנתונים כדי להעלות')}
      </a>
    );
  }
  if (remedy.kind === 'supply') {
    const code = String((remedy.block || {}).code || '');
    if (onOpenCampaign && OPENS_THE_CAMPAIGN.indexOf(code) >= 0) {
      return (
        <Button type="button" className="pacing-remedy" onClick={onOpenCampaign}>
          {pick(locale, 'Open the campaign', 'פתחו את הקמפיין')}
        </Button>
      );
    }
    if (OPENS_THE_UPLOAD.indexOf(code) >= 0) {
      return (
        <a className="pacing-remedy" href={UPLOAD_HASH}>
          <Upload size={13} aria-hidden="true" />
          {pick(locale, 'Open Data to upload it', 'פתחו את מסך הנתונים כדי להעלות')}
        </a>
      );
    }
  }
  // The sentence a supply remedy states is the same block the row already prints
  // above the track, where a reader meets it before anything else on the row.
  // Printing it again in the control slot said the same thing twice, which reads
  // as two problems rather than as one.
  return null;
}

// The other ending. A row the board is asking a decision about is finished with
// either by acting on it or by somebody recording that the risk stands, and the
// second one is the only ending available on every such row. Once it is recorded
// the row states it, so a person scanning the board can see at a glance which
// rows have been read and which have not.
export function Acceptance({ acceptance, locale, busy, onAccept, onOpenLedger }) {
  if (!acceptance || acceptance.kind === 'none') return null;
  if (acceptance.kind === 'accepted') {
    return (
      <Button type="button" className="pacing-accepted" onClick={() => onOpenLedger(acceptance.makeGoodId)}>
        <ShieldCheck size={13} aria-hidden="true" />
        {pick(locale, 'Risk taken on, open the record', 'הסיכון התקבל, פתחו את הרשומה')}
      </Button>
    );
  }
  return (
    <Button type="button" className="pacing-accept" disabled={busy} onClick={onAccept}>
      {pick(locale, 'Take the risk on', 'קבלו את הסיכון')}
    </Button>
  );
}
