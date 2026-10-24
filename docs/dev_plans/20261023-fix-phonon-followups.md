# Task: tts-server — phonon post-merge follow-ups

**Status**: In Review — implemented on `fix/phonon-followups`
**Component**: tts-server (backends, ops scripts, example adapter)
**Assigned to**: Varun Singh
**Priority**: Low
**Branch**: `fix/phonon-followups` (off `main` @ `802914b`)
**Created**: 2026-10-23
**Review Gates**: none

## Objective

Close the four gaps left open after the phonon backend merged (PR #18). Every change is
backwards compatible with wire protocol v0.1; none touches another backend.

## Decisions (grilled 2026-10-23)

| # | Item | Outcome |
|---|---|---|
| 1 | No server-side sentence splitter: a sentence over ~45 s of audio fails the response (`PhononTruncationError`) | **Waived.** The tripwire already makes the failure explicit and the README tells clients to send short sentences. A single sentence over ~150 words is rare in sentence-aggregated Pipecat output (a guess; no production data). A splitter would change prosody at every split and set a precedent for text rewriting. A phonon-local and a shared-helper variant were both considered. |
| 2 | Reference adapter does not retry on `busy` | **Accepted.** Re-send only the commit after `retry_after_ms` (+ up to 25% jitter); defaults `busy_retries=3`, `busy_retry_budget_s=3.0`; `busy_retries=0` restores the old drop behaviour; no usable hint or a non-`busy` code is never retried; a barge-in during the wait cancels and clears. |
| 3 | Stale quantize temp directory after a crash | **Accepted.** At quantized-alias start, remove only real directories named `.<name>.tmp-<int>-<int>` whose newest mtime is over 10 minutes old (first proposed at 1 h, shortened because a convert killed on every start leaves one ~60 MB directory per launchd restart, so the age bounds the pile-up: ~60 directories instead of ~360); age, not pid liveness (a pid check is wrong across hosts sharing a home directory and after pid reuse). |
| 4 | Plist regexes use `re.match` with `^…$` | **Fact, no decision.** `$` matches before a trailing newline; all validated values now use `fullmatch`. |

## Facts verified

- The server leaves the text buffer intact when it rejects a commit with `busy`; it clears
  it only after the scheduler admits the commit (`tts_server/server.py`).
- `docs/protocol.md` already recommends a bounded client retry on `busy`.
- `pocket_tts` does not share phonon's 45 s limit: one 250-word run-on returned the full
  91.1 s (single run). `qwen3_tts` and `fish_tts` have their own per-generation token
  tripwires (`_truncation_util.py`).

## Files to Modify

- `scripts/render_tts_plist.py`, `tests/test_render_tts_plist.py`
- `tts_server/backends/phonon.py`, `tests/test_phonon_lean.py`
- `examples/pipecat_tts_service.py`, `tests/test_pipecat_adapter.py`
- `CHANGELOG.md` (under `[Unreleased]`), `README.md`

## Implementation Checklist

### Phase 1: Follow-ups

- [x] Plist: `fullmatch` for every validated value; one trailing-newline case per variable (fails on the old script, passes on the new)
- [x] Phonon: age-based sweep of stale quantize temp directories
- [x] Adapter: bounded `busy` retry with opt-out
- [x] CHANGELOG `[Unreleased]` and README

## Open

- The retry budget (3.0 s) and the sweep age (10 min) are chosen, not measured.
- No live-server test of the busy retry; coverage is unit-level against a scripted client, plus the server-code check above.
