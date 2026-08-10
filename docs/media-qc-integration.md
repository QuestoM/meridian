# Media QC integration

P13 consumes measurements made by the system that actually receives or
transcodes broadcast masters. Meridian does not accept a manual "verified"
button and does not infer file health from the traffic booking.

## Data flow

```text
MAM / delivery inbox / ingest / transcoder / QC
  -> CSV report keyed by House Number
  -> scripts/import_media_report.py
  -> data/media_assets.csv
  -> kairos_api.media_verdict
  -> pod payload, row verdict and server-side lock gate
```

The integration owner chooses the real upstream system and schedules the import.
The importer is atomic and idempotent by House Number. A report with a missing
or duplicate House Number, or with no measured fact, is refused before the
canonical store is replaced.

## Canonical report contract

The header in `data/media_assets.csv` is authoritative. It carries exact seconds
and frames, frame rate, container and codec, pixel dimensions and display aspect,
audio presence and layout, loudness and its standard, approval provenance, the
measurement clock and the source. Legacy feed names `creative_id`, `aspect_ratio`,
`has_audio`, `codec`, `audio_channels` and `qc_state` are accepted as import
aliases, but Meridian stores the canonical names.

Example invocation after a real report has landed:

```bash
python scripts/import_media_report.py /path/from/the/qc-system/report.csv \
  --source "<real system and report identity>"
```

`config/media_standards.json` is a separate owner-supplied input. It defines the
accepted containers, codecs, rates, dimensions, audio layouts, loudness target
and approval vocabulary. It deliberately ships empty. A measurement without a
corresponding standard is `unavailable`, never an implicit pass.

## Lock rule

An unmeasured or incompletely configured asset is not cleared and is not called
broken, so it is shown as unavailable. A measured mismatch or explicit rejected
approval is failed. The dashboard disables lock for that pod and the API rejects
the lock independently with HTTP 409, so a direct request cannot bypass the
surface.
