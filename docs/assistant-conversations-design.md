# Assistant conversations: design

Owner ask (2026-07-28): the assistant should hold multiple conversations, not one
endless thread: a conversation list with history, per-conversation visibility of
the changes applied during it, conversation-level restore of those changes, and
the acting user displayed.

Ground truth this design is built on (recon 2026-07-28, all measured):

- The thread store is one flat JSON per user (`data/assistant/threads/<user>.json`),
  entries carry exactly `{question, answer, at, batch_id}` and are pruned to 50
  (`kairos_api/assistant_memory.py`). Real store: 1 user, 8 entries.
- Every ask that produced proposals already stores its `batch_id` on the entry
  (both ask paths); batches carry `created_by`, model, question, and per-item
  `status/resolved_by/resolved_at` (`kairos_api/assistant_actions.py`).
- Two restore primitives exist and are keyed by batch_id: the assistant restore
  points (pre-apply copies of exactly the touched files) and the unified version
  timeline (`version_store.snapshot_assistant_apply`, source `assistant_apply`,
  batch_id in the manifest, restore is itself undoable via `pre_restore`).
- The frontend drops the stored batch_id when loading saved history
  (`AssistantPanel.jsx` maps `batchId: null`), which already breaks the
  proposal-card linkage across reloads. The chat UI displays no acting user.
- `assistant_history.py` replays the whole user thread into the model (newest 6
  exchanges, 12k chars), so without scoping, conversations would cross-contaminate.

## Decisions

1. Conversation entity. A conversation is a named sequence of exchanges with an
   id. Storage moves to one file per conversation under
   `data/assistant/threads/<user>/<conversation_id>.json` plus a small per-user
   index file (`index.json`: id, title, created_at, updated_at, entry_count,
   user). Ids are `uuid4().hex[:12]` minted server-side on the first ask of a
   new conversation. Title defaults to the first question truncated to 60 chars;
   renameable.
2. Schema. Thread entries gain `conversation_id`; `_ENTRY_KEYS` grows in the
   same change that starts writing it (the loader projection strips unknown
   keys, so ordering matters). Batches gain `conversation_id` through
   `create_batch`. Version manifests need NO change: conversation to batches to
   versions resolves through the existing batch_id.
3. API. `GET /api/assistant/conversations` (index, newest first),
   `POST /api/assistant/conversations` (new, returns id),
   `PATCH .../{id}` (rename), `DELETE .../{id}` (that conversation only, audited),
   `GET /api/assistant/thread?conversation_id=...` (entries of one conversation;
   without the param, the newest conversation, keeping the old client working).
   `AskRequest` gains optional `conversation_id`; absent means the client asks in
   the active conversation it names, a missing or unknown id mints a new one and
   the response carries the id back.
4. History scoping. `history_messages(username, conversation_id)` windows only
   the active conversation. The 6-exchange and 12k-char caps stay.
5. Applied-changes view. Per conversation, the UI lists its batches (filter
   proposals by conversation_id, with a batch_id-set fallback for the legacy
   conversation), each with kind, summary, status, resolved_by, resolved_at, and
   a link into the matching version diff on the restore page.
6. Conversation-level restore. A thin orchestration over shipped primitives:
   collect the conversation's batch_ids with any applied item; list
   `assistant_apply` versions with those batch_ids; for each logical file, pick
   the OLDEST such version (its snapshot is the state BEFORE the conversation's
   first mutation of that file) and restore it via the existing per-version
   restore, which snapshots `pre_restore` first so the whole operation is
   undoable. Honest limits, stated in the confirm dialog: recomputes cannot be
   un-run (inputs are restored, then a recompute is offered), and a whole-file
   restore also reverts manual edits made to the same file after the
   conversation (interleaving is the norm: 94 of 97 real versions are manual
   edits). The byte-identical snapshot short-circuit means a batch may lack its
   own version; the mapping uses the nearest older `assistant_apply` version.
7. Acting user. The chat header shows the authenticated user (`/thread` already
   returns it); each batch row shows `created_by` and `resolved_by`. When auth
   is disabled the actor is honestly `auth-disabled`, displayed as-is.
8. Migration. On first read of a legacy flat `threads/<user>.json`, wrap its
   entries as one conversation titled 'שיחה קודמת' with id `legacy-<user>`,
   write via the existing atomic path, and delete the flat file only after the
   new files are written. Batches without conversation_id display under the
   legacy conversation when their batch_id matches one of its entries.
9. Caps. Per-conversation entry cap stays 50; the index caps at 30 conversations
   per user (oldest pruned with their files, disclosed in the UI).

## Non-goals

Cross-user conversation sharing; server-side conversation search; editing past
exchanges; per-message restore granularity (restore is per conversation, through
whole-file versions, as the version store works today).
