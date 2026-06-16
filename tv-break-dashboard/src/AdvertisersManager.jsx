import React, { useCallback, useEffect, useState } from 'react';
import { Button, FormControlLabel, Switch, TextField } from '@mui/material';
import { Plus, RefreshCcw, Save, Trash2, Users } from 'lucide-react';

const API_BASE = import.meta.env.VITE_KAIROS_API_URL || 'http://127.0.0.1:8000';

function pageText(locale, en, he) {
  return locale === 'he' ? he : en;
}

function normalizeRows(value) {
  return Array.isArray(value) ? value : [];
}

const emptyAdvertiser = {
  advertiser_id: '',
  default_premium: 0,
  allow_positions: '',
  allow_genres: '',
  prime_time_only: false,
  notes: '',
};

function AdvertiserRow({ row, locale, onSave, onDelete }) {
  const [draft, setDraft] = useState(row);
  const [saving, setSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    setDraft(row);
  }, [row]);

  function update(field, value) {
    setDraft((current) => ({ ...current, [field]: value }));
  }

  async function handleSave() {
    setSaving(true);
    await onSave(draft);
    setSaving(false);
  }

  return (
    <div className="advertiser-row">
      <div className="advertiser-key" dir="ltr">{row.advertiser_id}</div>
      <div className="advertiser-fields">
        <TextField
          label={pageText(locale, 'Default premium', 'תוספת ברירת מחדל')}
          type="number"
          size="small"
          value={draft.default_premium ?? 0}
          onChange={(event) => update('default_premium', Number(event.target.value))}
        />
        <TextField
          label={pageText(locale, 'Allowed positions', 'מיקומים מותרים')}
          size="small"
          value={draft.allow_positions || ''}
          onChange={(event) => update('allow_positions', event.target.value)}
        />
        <TextField
          label={pageText(locale, 'Allowed genres', 'ז׳אנרים מותרים')}
          size="small"
          value={draft.allow_genres || ''}
          onChange={(event) => update('allow_genres', event.target.value)}
        />
        <TextField
          label={pageText(locale, 'Notes', 'הערות')}
          size="small"
          value={draft.notes || ''}
          onChange={(event) => update('notes', event.target.value)}
        />
        <FormControlLabel
          className="advertiser-switch"
          control={
            <Switch
              size="small"
              checked={Boolean(draft.prime_time_only)}
              onChange={(event) => update('prime_time_only', event.target.checked)}
            />
          }
          label={pageText(locale, 'Prime time only', 'פריים טיים בלבד')}
        />
      </div>
      <div className="advertiser-actions">
        <Button className="secondary-button compact" type="button" variant="outlined" disabled={saving} onClick={handleSave}>
          <Save size={14} />
          {saving ? pageText(locale, 'Saving...', 'שומר...') : pageText(locale, 'Save', 'שמירה')}
        </Button>
        {confirmDelete ? (
          <Button className="secondary-button compact danger" type="button" variant="outlined" onClick={() => onDelete(row.advertiser_id)}>
            <Trash2 size={14} />
            {pageText(locale, 'Confirm', 'אישור')}
          </Button>
        ) : (
          <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setConfirmDelete(true)}>
            <Trash2 size={14} />
            {pageText(locale, 'Delete?', 'מחיקה?')}
          </Button>
        )}
        {confirmDelete && (
          <Button className="secondary-button compact" type="button" variant="outlined" onClick={() => setConfirmDelete(false)}>
            {pageText(locale, 'Cancel', 'ביטול')}
          </Button>
        )}
      </div>
    </div>
  );
}

function AddAdvertiser({ locale, onCreate }) {
  const [draft, setDraft] = useState(emptyAdvertiser);
  const [creating, setCreating] = useState(false);

  function update(field, value) {
    setDraft((current) => ({ ...current, [field]: value }));
  }

  async function handleCreate() {
    if (!draft.advertiser_id.trim()) {
      return;
    }
    setCreating(true);
    const ok = await onCreate(draft);
    setCreating(false);
    if (ok) {
      setDraft(emptyAdvertiser);
    }
  }

  return (
    <div className="advertiser-add">
      <TextField
        label={pageText(locale, 'Advertiser ID', 'מזהה מפרסם')}
        size="small"
        value={draft.advertiser_id}
        onChange={(event) => update('advertiser_id', event.target.value)}
      />
      <TextField
        label={pageText(locale, 'Default premium', 'תוספת ברירת מחדל')}
        type="number"
        size="small"
        value={draft.default_premium}
        onChange={(event) => update('default_premium', Number(event.target.value))}
      />
      <TextField
        label={pageText(locale, 'Allowed positions', 'מיקומים מותרים')}
        size="small"
        value={draft.allow_positions}
        onChange={(event) => update('allow_positions', event.target.value)}
      />
      <TextField
        label={pageText(locale, 'Allowed genres', 'ז׳אנרים מותרים')}
        size="small"
        value={draft.allow_genres}
        onChange={(event) => update('allow_genres', event.target.value)}
      />
      <FormControlLabel
        className="advertiser-switch"
        control={
          <Switch
            size="small"
            checked={Boolean(draft.prime_time_only)}
            onChange={(event) => update('prime_time_only', event.target.checked)}
          />
        }
        label={pageText(locale, 'Prime time only', 'פריים טיים בלבד')}
      />
      <Button className="run-button" type="button" variant="contained" disabled={creating || !draft.advertiser_id.trim()} onClick={handleCreate}>
        <Plus size={15} />
        {creating ? pageText(locale, 'Adding...', 'מוסיף...') : pageText(locale, 'Add advertiser', 'הוספת מפרסם')}
      </Button>
    </div>
  );
}

function AdvertisersManager({ copy, locale, notify }) {
  const [advertisers, setAdvertisers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState(true);

  const loadAdvertisers = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/advertisers`);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      const payload = await response.json();
      setAdvertisers(normalizeRows(payload.advertisers));
      setOnline(true);
    } catch {
      setAdvertisers([]);
      setOnline(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAdvertisers();
  }, [loadAdvertisers]);

  async function handleSave(draft) {
    try {
      const response = await fetch(`${API_BASE}/api/advertisers/${encodeURIComponent(draft.advertiser_id)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(draft),
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Advertiser ${draft.advertiser_id} saved.`, `המפרסם ${draft.advertiser_id} נשמר.`);
      await loadAdvertisers();
    } catch (error) {
      notify(`Save failed for ${draft.advertiser_id} (${error.message}).`, `השמירה נכשלה עבור ${draft.advertiser_id} (${error.message}).`);
    }
  }

  async function handleCreate(draft) {
    try {
      const response = await fetch(`${API_BASE}/api/advertisers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(draft),
      });
      if (response.status === 409) {
        notify(`Advertiser ${draft.advertiser_id} already exists.`, `המפרסם ${draft.advertiser_id} כבר קיים.`);
        return false;
      }
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Advertiser ${draft.advertiser_id} created.`, `המפרסם ${draft.advertiser_id} נוצר.`);
      await loadAdvertisers();
      return true;
    } catch (error) {
      notify(`Create failed (${error.message}).`, `היצירה נכשלה (${error.message}).`);
      return false;
    }
  }

  async function handleDelete(advertiserId) {
    try {
      const response = await fetch(`${API_BASE}/api/advertisers/${encodeURIComponent(advertiserId)}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      notify(`Advertiser ${advertiserId} deleted.`, `המפרסם ${advertiserId} נמחק.`);
      await loadAdvertisers();
    } catch (error) {
      notify(`Delete failed for ${advertiserId} (${error.message}).`, `המחיקה נכשלה עבור ${advertiserId} (${error.message}).`);
    }
  }

  return (
    <section className="page-workspace">
      <div className="page-header">
        <div>
          <h1>{pageText(locale, 'Advertisers', 'מפרסמים')}</h1>
          <p>
            {pageText(
              locale,
              'Manage advertiser rules: default premium, allowed positions and genres, prime-time limits, and notes.',
              'ניהול תנאי מפרסמים: תוספת ברירת מחדל, מיקומים וז׳אנרים מותרים, מגבלות פריים טיים והערות.',
            )}
          </p>
        </div>
        <Button className="secondary-button compact" type="button" variant="outlined" onClick={loadAdvertisers}>
          <RefreshCcw size={14} />
          {copy?.refresh || pageText(locale, 'Refresh', 'רענון')}
        </Button>
      </div>

      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Add advertiser', 'הוספת מפרסם')}</h2>
          <span><Users size={14} /></span>
        </div>
        <div className="advertiser-add-wrap">
          <AddAdvertiser locale={locale} onCreate={handleCreate} />
        </div>
      </section>

      <section className="page-panel">
        <div className="panel-head">
          <h2>{pageText(locale, 'Advertiser rules', 'תנאי מפרסמים')}</h2>
          <span>{advertisers.length} {pageText(locale, 'advertisers', 'מפרסמים')}</span>
        </div>
        {loading && <div className="upload-state">{pageText(locale, 'Loading advertisers...', 'טוען מפרסמים...')}</div>}
        {!loading && !online && (
          <div className="upload-state error">{pageText(locale, 'The Kairos API is unavailable. Advertisers cannot be shown.', 'ה־API של Kairos לא זמין. לא ניתן להציג מפרסמים.')}</div>
        )}
        {!loading && online && advertisers.length === 0 && (
          <div className="upload-state">{pageText(locale, 'No advertiser rules were found.', 'לא נמצאו תנאי מפרסמים.')}</div>
        )}
        {!loading && online && advertisers.length > 0 && (
          <div className="advertiser-list">
            {advertisers.map((row) => (
              <AdvertiserRow
                key={row.advertiser_id}
                row={row}
                locale={locale}
                onSave={handleSave}
                onDelete={handleDelete}
              />
            ))}
          </div>
        )}
      </section>
    </section>
  );
}

export default AdvertisersManager;
