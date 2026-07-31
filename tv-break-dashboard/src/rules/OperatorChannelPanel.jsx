import React from 'react';
import { FormControl, MenuItem, Select } from '@mui/material';
import { Tv } from 'lucide-react';
import { normalizeRows } from '../shell/plan-model';

// OperatorChannelPanel: shows available_channels from /api/parameters and lets
// the operator choose which channel they own. The selection is persisted via
// the same PUT /api/settings path as all other settings.
export function OperatorChannelPanel({ settings, parameters, locale, onSave, saveState, featured }) {
  const he = locale === 'he';
  const availableChannels = normalizeRows(
    parameters?.available_channels || parameters?.settings?.available_channels,
  );
  const currentChannel = settings?.operator_channel || '';

  function handleChange(channel) {
    onSave({ ...settings, operator_channel: channel });
  }

  return (
    <section className={`settings-panel wide${featured ? ' settings-panel-featured' : ''}`}>
      <div className="settings-panel-head">
        <div>
          {featured && (
            <span className="settings-channel-kicker">{he ? 'נקודת הפתיחה' : 'Start here'}</span>
          )}
          <h2>{he ? 'הערוץ שלכם' : 'Your channel'}</h2>
          <p>{he ? 'הערוץ שבבעלות המפעיל. האילוצים שלכם חלים על ערוץ זה, ותחזית ההכנסה מול שימור הצופים מחושבת עבורו.' : 'The channel this operator owns. Your constraints apply to this channel, and the revenue versus retention forecast is computed for it.'}</p>
        </div>
        <Tv size={18} />
      </div>
      <label htmlFor="operator-channel-select" style={{ display: 'block', marginBottom: 6, fontSize: 12, fontWeight: 600, color: 'var(--muted)' }}>
        {he ? 'ערוץ' : 'Channel'}
      </label>
      <FormControl size="small" sx={{ minWidth: 220 }}>
        <Select
          id="operator-channel-select"
          value={currentChannel}
          displayEmpty
          onChange={(e) => handleChange(e.target.value)}
          renderValue={(selected) => selected || (he ? 'לא נבחר' : 'Not set')}
        >
          <MenuItem value="">{he ? 'לא נבחר' : 'Not set'}</MenuItem>
          {availableChannels.map((ch) => {
            const val = typeof ch === 'string' ? ch : ch.key || ch.value || ch.name || String(ch);
            return <MenuItem key={val} value={val}>{val}</MenuItem>;
          })}
        </Select>
      </FormControl>
      {currentChannel && (
        <p className="cb-operator-channel-note">
          {he ? `האילוצים החדשים יחולו על ערוץ "${currentChannel}".` : `New constraints will be scoped to channel "${currentChannel}".`}
        </p>
      )}
      {!currentChannel && (
        <p className="cb-operator-channel-warning">
          {he ? 'אזהרה: הערוץ אינו מוגדר. מסנן הערוץ המתחרה אינו פעיל - האילוצים חלים על כל הערוצים עד שתבחרו ערוץ.' : 'Warning: no channel is set. The competitor-channel boundary filter is inactive - constraints match all channels until you pick your channel.'}
        </p>
      )}
    </section>
  );
}

export default OperatorChannelPanel;
