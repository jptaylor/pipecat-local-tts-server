# Task: tts-server — `phonon` (Gradium Phonon 31M) backend

**Status**: Complete — implementation and review gauntlet done on `feature/tts-phonon-backend` (PR #18)
**Component**: tts-server (backends)
**Assigned to**: Varun Singh
**Priority**: Medium
**Branch**: `feature/tts-phonon-backend` (off `main` @ `a3f1192`)
**Created**: 2026-10-21
**Review Gates**: full

## Objective

Add a `phonon` streaming TTS backend (`gradium/phonon-31m-en`; licence verified in Phase 0 Q6,
the spike saw Apache-2.0) through mlx-audio 0.6.1's `pocket_tts` module, with **operator-selected
named voices** drawn from the ~102-voice `gradium/phonon-voices` catalog, a `voices` CLI
subcommand to list that catalog, unit tests, justfile smoke recipes, a profiling entry, and a
**0.7.0** release. **No session-loop or wire-protocol changes** — the backend plugs into the
existing `BridgedStream` (`tts_server/backends/_bridge_stream.py`) / `_stream_util` bridge like
every other backend.

## Context

- **Spike (2026-10-21, scratchpad `phonon_spike.py`, mlx-audio 0.6.1, rev `52433c4697`)**:
  load 1.2 s, TTFB 43 ms, RTF 0.034, 24 kHz, default chunks 2.0 s (`streaming_interval=2.0`).
  With Parakeet-v3 looping in another process: TTFB 53 ms, RTF 0.042. `gen.close()` between
  chunks cancels cleanly and the model stays healthy. Details in memory `phonon-tts-spike.md`.
- **Loader**: Phonon has no module of its own; its checkpoint (`flow_lm` + `mimi` keys, no
  `model_type`) routes to `mlx_audio.tts.models.pocket_tts` (PR #1049, first in 0.6.1). The pin
  is already `mlx-audio==0.6.1` everywhere (PR #17 / 0.6.0), so **no pin conflict remains**.
- **Blocker**: on 2026-10-21 Gradium pushed HF commit `5c8a65f0` ("add transformers support")
  setting `model_type: gradium_phonon_tts`. mlx-audio 0.6.1 and main (`7cc680b2`) fail with
  "Model type gradium_phonon_tts not supported". Revision `52433c4697` loads. The backend must
  therefore load a **pinned revision** until a separate repo-wide mlx-audio bump supports the
  new config. Revision durability (the sha staying fetchable and ungated) is an **assumption**;
  Phase 0 Q1 is its point-in-time detector and the recovery path is `--model <absolute snapshot
  path>`, which loads as given (see Requirements).
- **Voices (review-plan reads of the live repos, 2026-10-21)**: the 31M checkpoint has only
  `default` (shipped as `default-voice.safetensors`) and no speaker encoder (`ref_audio` raises
  "checkpoint omits the speaker encoder"). `gradium/phonon-voices` holds **102 voices** —
  37 English, 15 French, 13 German, 25 Spanish, 12 Portuguese — as
  `voices/<id>.safetensors` (1×512×125, ~256 KB each, ~26 MB total), plus 102 `samples/*.ogg`
  and a machine-readable `voices.jsonl` (name, voice_id, language, locale, gender, age,
  description). Both repos read as Apache-2.0 / ungated with an authenticated client — Phase 0
  re-verifies with `token=False`. The voices repo (updated 2026-10-20, format tag
  `ba7a5a45@300`) is newer than the pinned model revision, and only one voice was tried in the
  spike. Display names are not plain ASCII slugs (e.g. "Elodie-Rose", "Lucía-Sol", "Captain
  Silas Blackwater"). Named-voice TTFB ≈ 0.14 s vs 0.043 s default (mlx-audio recomputes the
  voice prompt state on every `generate()` call, so the cost is per call); first fetch ≈ 1.2 s.
- **Voice security constraint**: `server._validate_voice` restricts `voice` to the advertised
  set because mlx-audio treats `*.safetensors` / `hf://` strings as loadable paths (file read /
  network egress). For the encoder-less 31M an unknown bare name raises `ValueError("Unknown
  voice")` in mlx-audio 0.6.1, but the loader still special-cases bare names for other models
  (e.g. Kyutai's `PREDEFINED_VOICES`, and the catalog has a voice named "Alba"). Path-only
  resolution is therefore kept as defence in depth: the resolved argument is **always an
  absolute local `.safetensors` path**.
- **Operator settings precedent**: there is no runtime config-file mechanism (the only `toml`
  is `pyproject.toml`). The one backend-specific operator setting so far,
  `PIPECAT_TTS_KOKORO_EXTRA_LANGS`, is env-only: read inside the backend constructor
  (`kokoro.py`, `env_str_set`), baked into the launchd plist and syntax-validated by
  `scripts/render_tts_plist.py`, and an unknown value is **warned about and dropped**, not
  fatal. Voice selection follows that precedent exactly.
- **HF cache under launchd is not new**: every backend already loads model weights from HF with
  only `HOME` baked into the plist, so the default `~/.cache/huggingface` is what agents use.
  This plan adds no HF environment passthrough.
- 40M Phonon models were recorded in the spike as CC BY-NC and gated — out of scope (not
  defaulted, not smoke-tested); no doc quotes that statement unless Phase 0 Q6 verifies it.
- `docs/dev_plans/README.md` does not exist; nothing to index.

## Requirements

- **Backend `phonon`**: `streaming: true`, 24 kHz, `languages: ["en"]`, default model
  `gradium/phonon-31m-en`.
- **Revision pin = per-repo table** `{repo_id: sha}` (full 40-char sha from Phase 0 Q2). A model
  with a table entry is fetched with `snapshot_download(repo, revision=sha)` (allow-patterns
  chosen from the file list Q2 records, mirroring mlx-audio's `DEFAULT_ALLOW_PATTERNS`) and
  loaded from the returned local path. A model with no entry (another repo id, or a local path
  such as an absolute snapshot path) loads as given with a logged warning — this is also the
  recovery path if the pinned sha disappears. The effective repo, revision and resolved path
  are logged at startup. An unreachable or unknown pinned sha is a **startup error** — never a
  fallback to HEAD.
- **Voices — catalog vs enabled set**:
  - The *catalog* is a static name→id table bundled in the package, derived from `voices.jsonl`
    at a recorded 40-char voices-repo sha, with a language per voice. Names are slugged
    (NFKD-strip diacritics, lowercase, non-alphanumerics → `_`) and unique after slugging.
  - The *enabled set* is chosen by the operator ahead of time with env var
    `PIPECAT_TTS_PHONON_VOICES` (comma-separated slugs, matching
    `^[a-z0-9_]+(,[a-z0-9_]+)*$`; duplicates collapsed in order), read in
    `PhononBackend.__init__` with the same helper Kokoro uses. Unset → `default` only. There is
    **no `serve` flag** and no change to `make_backend`.
  - Only enabled voices are advertised (`default` always first) and only enabled voices are
    downloaded, **one file each**: `hf_hub_download(voices_repo, "voices/<id>.safetensors",
    revision=<voices sha>)`, trying `local_files_only=True` first so a warm cache makes no
    network call. After download the backend asserts `os.path.isfile` for every resolved path.
  - **Soft failure (Kokoro-style)**: an enabled name outside the catalog, or an enabled voice
    that cannot be fetched (offline and uncached, gated, deleted), is logged as a warning and
    **skipped** — not advertised, server still starts with `default` plus the voices that
    resolved. A client asking for a skipped voice gets `invalid_config` from the existing server
    validation; there is no silent substitution of the default voice. The final advertised list
    is logged at startup and is what `server.status` reports.
  - Enabling a non-English voice logs a warning (the model is `-en`); Phase 0 Q4 synthesizes
    several to document the caveat.
  - `generate(voice=...)` receives only an absolute local `.safetensors` path (`default` →
    kwarg omitted); no client string, `hf://` URL or bare name ever reaches it.
- **Catalog CLI**: `python -m tts_server voices [--backend phonon]` lists the catalog (name,
  language, enabled, cached); `--backend` defaults to `phonon` and any other value exits non-zero
  with "no voice catalog for backend X". `--refresh` fetches the current `voices.jsonl`, shows
  what differs from the bundled table and **changes nothing** (v1). Using refreshed voices means
  updating the bundled table after verification against the pinned model. `voices` is added to
  the subcommand dispatch set in `__main__.py`. No wire-protocol surface.
- Never forward `ref_audio` or `frames_after_eos` (no cloning — v1 decision #2; same negative
  guard as `pocket_tts`). Extras = what `generate()` accepts (Phase 0 Q3; expected
  `temperature`); enforcement is the `merge_extras(_EXTRA_COERCERS, …)` allowlist.
  `_introspect_util.verify_generate_signature` additionally warns at serve time (log-only).
- `phonon` extra in `pyproject.toml`: `websockets>=13.0`, `mlx-audio==0.6.1`, `requests`
  (undeclared by 0.6.1's `pocket_tts/utils.py`, same as the `pocket_tts` extra) and
  `huggingface_hub>=2.0` (imported directly, lazily, inside `start()`/CLI handlers only; the
  offline-cache behaviour relied on here is a 2.x behaviour, and Q2 records the locked
  version). Lean base stays import-clean.
- Cancel/barge-in via the existing `BridgedStream.cancel()` — verified between chunks and
  through the real server.
- The repo-wide mlx-audio pin stays `0.6.1`; moving it is a separate bump.
- Version `0.6.0` → `0.7.0`, CHANGELOG entry, README/AGENTS/protocol docs, release by the user
  after merge.
- No upstream-facing action (issue/PR to mlx-audio) without the user's say-so; this plan only
  drafts the note.

## Review Focus

- **Revision pinning**: the pin is applied by `snapshot_download(revision=…)` + load of the local
  path, scoped to repos in the pin table; a stale/unreachable pin fails startup loudly, never
  loads HEAD; non-table models are explicit (warning, loaded as given); revision durability is
  a named assumption with a recovery path.
- **Voice security**: no client string reaches `generate(voice=…)`; every resolved argument is
  an absolute local path that exists.
- **Soft-failure voice startup**: unknown or unfetchable enabled voices are skipped with a
  warning, never fatal, never silently replaced by the default voice.
- **Offline / first-use behaviour**: network only on a cache miss at startup, per enabled
  voice; offline+cached works; offline+uncached skips that voice.
- **Duplication vs `pocket_tts.py`**: decided — copy the small skeleton, leave `pocket_tts`
  untouched (see Architecture Decisions); revisit if a third pocket-family backend appears.
- **Upstream drift**: Gradium may re-upload again; the Phase 0 gate script must be re-runnable.
- **Operator surface**: `PIPECAT_TTS_PHONON_VOICES` validation (plist injection safety) and the
  `voices` subcommand.

## Implementation Checklist

Phase blocks are the immutable contract; progress lives in `## Progress` below the marker.

### Phase 0: Model verification gate (dia/qwen3/fish precedent)

**Impl files:** `tests/smoke/phonon_phase0_gate.py`
**Test files:** `tests/smoke/phonon_phase0_gate.py`
**Test command:** `uv run --extra pocket_tts python tests/smoke/phonon_phase0_gate.py`
**Goal:** Every assumption in this plan that the spike did not measure is answered by a run
against the installed 0.6.1 and the live HF repos (Q1 alone queries PyPI/upstream metadata
outside the pinned env), and recorded in `## Findings`; no Phase 1 code depends on a guess.
(`--extra pocket_tts` is used because the `phonon` extra does not exist until Phase 1; the two
extras differ only by the explicit `huggingface_hub` declaration, which mlx-audio already pulls
in transitively — Q2 asserts importability as the proof.)

- Q1 (informational, also the revision-durability detector): does HF HEAD of
  `gradium/phonon-31m-en` still fail with `gradium_phonon_tts`? Is the pinned sha still
  fetchable and ungated? Has a newer mlx-audio (PyPI release or main) added support? Run the
  upstream check outside the project env (e.g. `uv run --no-project --with <spec> python …`,
  or PyPI metadata plus a hand-recorded upstream sha). Recorded only; the repo pin stays
  `0.6.1` and a bump is a separate change.
- Q2: confirm the pin mechanism as an assertion — `snapshot_download(repo, revision=<40-char
  sha>)` returns a local snapshot, `load(local_path, lazy=False, strict=True)` loads it, a bogus
  sha raises, and `huggingface_hub` is importable from the dependency set. Record the full
  40-char model sha, the **file list and total size at that sha** (and the allow-patterns chosen
  from it), and the locked `huggingface_hub` version (must satisfy `>=2.0`; check how the
  offline/local-files-only path behaves on a complete vs a partial snapshot).
- Q3: `inspect.signature(model.generate)` — real kwargs; which extras to advertise; confirm
  `ref_audio` / `frames_after_eos` exist and are never forwarded; confirm `voice=None` omit
  rule; confirm `_introspect_util.verify_generate_signature` warns on drift (it is log-only —
  the `merge_extras` allowlist is the enforcement).
- Q4: voices — fetch `voices.jsonl` and record the voices-repo 40-char sha; record per-language
  counts and the slug rule's output (assert unique after slugging); verify both repos are
  ungated **unauthenticated**: `HfApi().model_info(repo, token=False)` with `not info.gated`
  (the field is `Literal["auto","manual",False] | None`) plus one `hf_hub_download(...,
  token=False)`; load **every** catalog `.safetensors` through the model's voice path at the
  pinned model revision before freezing the table, and **synthesize several English named
  voices, keeping the WAVs for a listen** (loading proves shape only); confirm `voice=` accepts
  an absolute local path; record what a bare unknown name does on this model (expected:
  `ValueError("Unknown voice")`), justifying path-only resolution as defence in depth; measure
  per-file download of N enabled voices (N = 1, 5, 37) vs the all-102 total, and named-voice
  TTFB with a warm cache; synthesize one non-English voice for the caveat.
- Q5: `streaming_interval` sweep (0.3 / 0.5 / 1.0 / 2.0 s) on TTFB and RTF. "Continuity" is
  measured **directly from `model.generate()` chunk timestamps** (no server exists in Phase 0):
  the maximum gap between consecutive chunk arrivals must not exceed the chunk duration. Lock
  one constant like `pocket_tts` (`_STREAMING_INTERVAL`). A server-level confirmation via
  `latency_smoke.py` happens in Phase 3.
- Q6: licence — fetch the model card and voices-repo licence text; docs quote only the verified
  text; record the 40M repos' licence/gating only if checked here (else drop the claim).
- Q7: cancel — close the generator at chunk boundaries (default and a named voice), then run a
  follow-up generate on the same model object. Server-level barge-in/disconnect is Phase 3.
- Q8: long-text probe — synthesize ~2000 characters, assert duration scales with word count
  (qwen3/fish found model-side truncation below the server cap); derive `_MAX_TEXT_CHARS` and
  `ideal_words` from the result, with a truncation tripwire only if truncation is observed.

### Phase 1: Backend module, voice catalog, unit tests (mlx-gated + lean)

**Impl files:** `tts_server/backends/phonon.py, tts_server/backends/phonon_voices.py, tts_server/backends/__init__.py, pyproject.toml, uv.lock, .github/workflows/test.yml`
**Test files:** `tests/test_phonon_lean.py, tests/test_phonon_backend.py, tests/test_import_safety.py`
**Test command:** `uv run --extra phonon pytest tests/test_phonon_lean.py tests/test_phonon_backend.py tests/test_import_safety.py -v` (pass condition: no skips — a skip means the mlx-gated coverage did not run)
**Goal:** `phonon.py` imports cleanly without mlx; with the extra installed it loads the pinned
revision from a local snapshot, advertises `default` + the enabled voices that resolved, and no
code path forwards a client string, `ref_audio`, or `frames_after_eos` to `generate()`.

- `phonon_voices.py` (lean, no mlx/hf imports at module load): bundled catalog table (slug →
  voice id, language), voices-repo sha, slug function, `resolve_enabled(names)` returning
  `(resolved, unknown)`.
- `phonon.py`: `PhononBackend(model=…)` (`backend_name = "phonon"`), reads
  `PIPECAT_TTS_PHONON_VOICES` in `__init__`; `DEFAULT_PHONON_MODEL`, pin table,
  `_PhononStream` (`BridgedStream` subclass, copied skeleton), `_STREAMING_INTERVAL` from Q5,
  `_MAX_TEXT_CHARS` from Q8. **Module-level patchable seams** `_snapshot_download`,
  `_load_model`, `_download_voice` (each lazy-imports its heavy dependency inside the call) so
  lean tests exercise `start()` without mlx or huggingface_hub; every existing lean test
  bypasses `start()` via `backend._loaded_model = spy`, which cannot cover the pin logic.
- `start()`: resolve the model (pin table → `_snapshot_download(revision=sha)` and
  `_load_model(local_path)`; else warn and load as given); read `sample_rate`; resolve enabled
  voices via `_download_voice` (local-files-only first, then network; skip-with-warning on
  failure; `os.path.isfile` assert on success); verify the `generate()` signature with
  `_introspect_util` (log-only); warm up under the metal lock; log effective revisions and the
  advertised voice list.
- Voice mapping: `default`/`None` → omit `voice` kwarg; named → absolute local path.
- `capabilities()`: `streaming: true`, `languages: ["en"]`, `voice_count`, `extras`, `ideal_words`,
  `max_text_chars` (values from Phase 0).
- `pyproject.toml`: `phonon` extra (incl. `huggingface_hub>=2.0`); regenerate `uv.lock`.
- Register in `make_backend` (signature unchanged: `make_backend(name, model=None)`); add **both**
  `phonon.py` and `phonon_voices.py` to the lean-module list in `test_import_safety.py`.
- CI (same commit as the lean test, per `test.yml` convention): add `tests/test_phonon_lean.py`
  to the lean allow-list and the `import tts_server.backends.phonon` + `DEFAULT_PHONON_MODEL`
  print to the macOS smoke block.
- Tests, lean (patching the module seams, no mlx): catalog integrity (slugs unique and
  `[a-z0-9_]+`, ids unique, count and per-language counts match Q4, `default` reserved);
  `resolve_enabled` unknown-name handling; env parsing (regex, duplicates, unset → default
  only); extras filtering via the `merge_extras` allowlist; forbidden kwargs never forwarded;
  pin spy tests (download carries the 40-char sha, `_load_model` receives the local snapshot
  path, a download exception propagates from `start()` with no fallback load call); non-table
  model → warning, no pin; **soft failure**: unknown enabled name and offline+uncached enabled
  voice are each skipped with a warning, server still starts, advertised list excludes them;
  warm cache → zero network-mode download calls (local-files-only tried first);
  `open_stream(voice="hf://…" | unknown name)` rejected at the backend layer; an enabled name
  (including `alba`, a real catalog slug) forwards an absolute `.safetensors` path and the
  literal slug never appears in `generate()` kwargs, while a not-enabled `alba` is rejected;
  `voice_count == len(voices())`; every resolved voice argument is an absolute existing
  `.safetensors` path; server-level check with a stub-loaded backend via `running_server`:
  status voices start with `default`, commit `voice="hf://x"` → `invalid_config`.
- Tests, mlx-gated: real load of the pinned revision (rate 24000, short synthesis), a named
  voice, cancel mid-stream then reuse, `HF_HUB_OFFLINE=1` start after the cache is warm, and a
  long-text regression (~`_MAX_TEXT_CHARS` input; seconds-per-word stays inside the band Q8
  recorded).

### Phase 2: Wiring — CLI, voice env var, justfile, smoke, plist renderer, docs, version

**Impl files:** `tts_server/__main__.py, justfile, scripts/render_tts_plist.py, scripts/install_tts_agent.sh, tests/smoke/run_smoke.sh, tests/smoke/run_multiconn.sh, .github/workflows/test.yml, README.md, AGENTS.md, CHANGELOG.md, docs/protocol.md, pyproject.toml, uv.lock, scripts/profiling/rtf_benchmark.py`
**Test files:** `tests/test_justfile_recipes.py, tests/test_render_tts_plist.py, tests/test_phonon_lean.py, tests/test_phonon_voices_cli.py`
**Test command:** `uv run --extra phonon pytest tests/test_justfile_recipes.py tests/test_render_tts_plist.py tests/test_phonon_lean.py tests/test_phonon_voices_cli.py -v`
**Validation cmd:** `just smoke-phonon`
**Goal:** Every enumeration of backend names in the repo includes `phonon`; the files that
`test_justfile_recipes.py` asserts equal land in ONE commit; a fresh checkout can
`uv sync --extra phonon && just smoke-phonon`.

- **Commit A (atomic — tests assert exact equality across these):** `__main__.py` `--backend`
  choices (~l.375) and default-model resolver (`_resolve_model`, ~l.35-66); `justfile`
  `_resolve` map (ports: tone 8665, kokoro 8765, voxtral 8865, pocket 8965, dia 9065, qwen3 9165,
  fish 9265 → **phonon 9365**), the backend-name case arm (~l.197), `smoke-phonon`,
  `smoke-multiconn-phonon`; README "Per-backend port convention" row
  `| phonon | pipecat.tts-server.phonon | 9365 |`; `render_tts_plist._BACKEND_RE` + usage string;
  `run_smoke.sh` / `run_multiconn.sh` guards; updated `test_justfile_recipes.py` /
  `test_render_tts_plist.py`; and the **dual-wire assertions in `test_phonon_lean.py`**
  (argparse choice and `_resolve_model` equality, per `test_fish_lean.py` precedent) — they
  cannot pass before `__main__.py` is edited, so they land here, not in Phase 1.
- **Commit B:** the plist renderer validates and bakes `PIPECAT_TTS_PHONON_VOICES` (strict regex
  above like the Kokoro one; hostile values rejected; baked only when the backend is `phonon`),
  `install_tts_agent.sh` passthrough/comment; the `voices` subcommand with `voices` added to
  the dispatch set in `main()` (and its docstring); `tests/test_phonon_voices_cli.py` — lean,
  isolated with `HF_HUB_CACHE=tmp_path` and a stubbed download seam serving a fake
  `voices.jsonl`; asserts the `cached` column flips when a fake `voices/<id>.safetensors` is
  placed, `--refresh` leaves the bundled catalog and the tmp cache byte-identical, no
  `--backend` defaults to phonon, a non-phonon backend exits non-zero; renderer tests
  `test_main_bakes_phonon_voices_into_plist`, `test_main_rejects_malformed_phonon_voices`, and
  env ignored with a warning on non-phonon backends; add `tests/test_phonon_voices_cli.py` to the
  `test.yml` lean allow-list.
- **Commit C:** docs and release prep — README (backends & licences table, install line, voice
  selection env var), AGENTS.md (`uv sync` line, CLI table incl. `voices`, env vars table,
  `--backend` choices, smoke recipes), `docs/protocol.md` (capabilities, voices, licence),
  CHANGELOG `## [0.7.0]`, `pyproject.toml` version `0.7.0` **with `uv.lock` regenerated in the
  same commit**, `rtf_benchmark.py` docstring only (add `phonon` to the backend list in its
  header; no code change), and `docs/upstream/mlx-audio-phonon-model-type.md` (draft note on the
  `gradium_phonon_tts` breakage and the pin; the user decides whether to file it).

### Phase 3: Smoke, profiling, adapter e2e

**Impl files:** `tests/smoke/reconnect_smoke.py, tests/smoke/latency_smoke.py, tests/smoke/phonon_cancel_smoke.py, tests/smoke/pipecat_adapter_e2e.py, scripts/profiling/README.md, tests/smoke/README.md, CHANGELOG.md`
**Test files:** `tests/test_pipecat_adapter.py`
**Test command:** `uv run --extra examples --extra phonon pytest tests/test_pipecat_adapter.py -v` (pass condition: no skips)
**Validation cmd:** `just smoke-phonon && just smoke-multiconn-phonon && uv run --extra phonon python tests/smoke/reconnect_smoke.py --backend phonon && uv run --extra phonon python tests/smoke/phonon_cancel_smoke.py && uv run --extra examples --extra phonon python tests/smoke/pipecat_adapter_e2e.py`
**Goal:** Measured numbers in the docs come from this repo's own harness against the shipped
backend, not from the spike, and barge-in/disconnect and a **named voice** are exercised through
the real server.

- `rtf_benchmark.py --backend phonon`: cold start, TTFB, RTF; default and one named voice; with
  and without a concurrent STT process (re-confirm the spike's contention numbers).
- `latency_smoke.py`: add a `max_gap` (maximum inter-chunk arrival gap) field to its output so
  the Phase 0 Q5 constant is confirmed at server level.
- `tests/smoke/phonon_cancel_smoke.py` (new, modelled on `dia_dialogue_smoke.py
  --check-disconnect`; confirm names at implementation): starts the server with
  `PIPECAT_TTS_PHONON_VOICES=<an English slug>,<a typo'd slug>`; asserts `server.status`
  voices == `["default", <slug>]` (typo skipped, server up), audio for `voice=<slug>`,
  `invalid_config` for `voice="hf://x"` and for the typo'd slug; then barge-in
  (`response.cancel` between chunks) and abrupt-disconnect-then-reuse.
- `reconnect_smoke.py`: extend its `--backend` choices (currently tone/kokoro) to `phonon`.
- `tests/smoke/pipecat_adapter_e2e.py` (new; promotes the ad-hoc 0.6.0 adapter e2e): uses the
  **default voice** and asserts on the **second utterance** (warm): 24 kHz, first audio frame
  TTFB < 0.1 s, ≥ 1 audio frame; the first utterance only has to succeed (named-voice TTFB is
  ≈ 0.14 s, so named voices are not asserted against 0.1 s).
- Record results in `scripts/profiling/README.md` and the CHANGELOG `0.7.0` entry; document the
  new smoke scripts in `tests/smoke/README.md`.

## Technical Specifications

### Files to Modify
- `tts_server/backends/__init__.py` — `make_backend` registration only (signature unchanged).
- `tts_server/__main__.py` — default model resolver, `--backend` choices, `voices` subcommand
  and the subcommand dispatch set.
- `pyproject.toml` — `phonon` extra; version `0.7.0`; extend the pin-policy comment.
- `uv.lock` — regenerated (Phase 1 for the extra, Phase 2 for the version).
- `justfile`, `scripts/render_tts_plist.py`, `scripts/install_tts_agent.sh`,
  `tests/smoke/run_smoke.sh`, `tests/smoke/run_multiconn.sh`, `tests/smoke/reconnect_smoke.py`,
  `tests/smoke/latency_smoke.py`, `.github/workflows/test.yml`.
- `tests/test_import_safety.py` (both new lean modules), `tests/test_justfile_recipes.py`
  (port 9365), `tests/test_render_tts_plist.py`.
- `README.md` (incl. port-convention table), `AGENTS.md`, `CHANGELOG.md`, `docs/protocol.md`,
  `scripts/profiling/rtf_benchmark.py` (docstring only), `scripts/profiling/README.md`,
  `tests/smoke/README.md`.

### New Files to Create
- `tts_server/backends/phonon.py` — the backend.
- `tts_server/backends/phonon_voices.py` — bundled voice catalog + slug/validation (lean).
- `tests/test_phonon_lean.py`, `tests/test_phonon_backend.py`, `tests/test_phonon_voices_cli.py`.
- `tests/smoke/phonon_phase0_gate.py` — re-runnable verification gate.
- `tests/smoke/phonon_cancel_smoke.py`, `tests/smoke/pipecat_adapter_e2e.py`.
- `docs/upstream/mlx-audio-phonon-model-type.md` — upstream note (draft only).

### Architecture Decisions
- Decision (grilled): **pin = per-repo table, applied by `snapshot_download(revision=sha)` then
  loading the local path.** Repos not in the table (other `--model` ids, local paths) load as
  given with a logged warning; the pin never silently disappears or mis-applies. No HEAD
  fallback exists anywhere in the backend; an unreachable pin is a startup error. The Q1
  "load HEAD with a newer mlx-audio" branch is dropped: the repo-wide pin moves only in a
  separate bump. Revision durability is a named assumption; recovery is
  `--model <absolute snapshot path>`.
- Decision (grilled): **voices are chosen ahead of time by the operator via the env var
  `PIPECAT_TTS_PHONON_VOICES` only** — consistent with the existing backend-specific setting
  (`PIPECAT_TTS_KOKORO_EXTRA_LANGS`: env-only, read in the backend, baked into the plist, no
  CLI flag, no `make_backend` change). Catalog (bundled, language-tagged, pinned voices sha) is
  separate from the enabled set (advertised + downloaded, one file each).
- Decision (grilled): **voice failures are soft, Kokoro-style.** An unknown or unfetchable
  enabled voice is warned about and skipped; `default` is always advertised; no silent
  substitution (a client asking for a skipped voice gets `invalid_config`). This removes the
  launchd KeepAlive restart-loop risk of a fail-closed typo and needs no installer-side
  catalog check.
- Decision (grilled): **no HF environment passthrough in the plist.** The HF cache location is
  an existing property of every backend (HOME-based default cache); this plan neither changes
  nor documents it beyond a note.
- Decision (grilled): **catalog listing/refresh is CLI-only, list-only in v1.** `voices
  --refresh` reports differences against `voices.jsonl` but changes nothing; adopting refreshed
  voices needs verification against the pinned model (the voices repo is newer than the pinned
  model revision). No wire-protocol surface: the protocol is frozen at v0.1, the server caches
  its voice set once in `start()`, and a client-triggered download is the egress path
  `_validate_voice` exists to close.
- Decision (grilled): **copy the small pocket skeleton; leave `pocket_tts.py` untouched.**
  `phonon.py` has its own `_PhononStream`/backend class but imports shared helpers
  (`BridgedStream`, `_extras_util`, `_introspect_util`). Accepted cost: duplication the repo's
  "lift on second use" notes warn about (memory `truncation-tripwire-duplicated-qwen3-fish`);
  revisit if a third pocket-family backend appears.
- **Test seams over `_loaded_model` bypass**: because pin/voice logic lives in `start()`,
  `phonon.py` exposes module-level `_snapshot_download` / `_load_model` / `_download_voice`
  seams; lean tests patch them (the existing `_loaded_model = spy` pattern cannot reach this
  code).
- **Static catalog, not runtime discovery**: names/ids fixed per pinned voices-repo sha; table
  integrity is unit-tested.
- **No cloning** (decision #2): `ref_audio` unwired; the 31M has no speaker encoder anyway.
- **Version**: new backend + new extra + new CLI surface = minor bump to 0.7.0; the user cuts
  the GitHub Release after merge (`pypi` environment needs manual deploy approval).

### Dependencies
- `mlx-audio==0.6.1` (pinned everywhere); `requests` (undeclared by 0.6.1);
  `huggingface_hub>=2.0` (declared in the `phonon` extra because it is imported directly and
  the cache semantics relied on are 2.x; mlx-audio itself only requires `>=1.0`; lazy import
  only).

### Integration Seams

| Seam | Writer | Caller | Contract |
|------|--------|--------|----------|
| voice name → generate arg | `phonon_voices` table + `PhononBackend` | `server._validate_voice` → `open_stream` | Only enabled, resolved slugs map; result is an absolute existing local `.safetensors` path or omitted (`default`); never `hf://`, http, or a bare name |
| enabled voices → plist | `render_tts_plist.py` | env `PIPECAT_TTS_PHONON_VOICES` | Validated by strict regex before baking, phonon backend only; hostile values rejected |
| cancel between chunks | `BridgedStream.cancel()` | server `response.cancel` | Generator closed at a yield boundary under the metal lock; model reusable immediately |
| pinned revision | `start()` via `_snapshot_download` / `_load_model` seams | `snapshot_download` → `mlx_audio.tts.utils.load(local_path)` | Table repos load only at their sha; failure to fetch is a startup error; non-table models warn |
| voice download | `start()` via `_download_voice` | `hf_hub_download` (per file) | local-files-only first; failure skips that voice with a warning; never on the commit path |
| atomic wiring set | Phase 2 commit A | `tests/test_justfile_recipes.py` | argparse choices == `just _resolve` == README port table == plist regex, in one commit; the `run_smoke.sh` / `run_multiconn.sh` guards and `install_tts_agent.sh` list are exercised by `just smoke-phonon` (Phase 2 validation), not by that test |

(Architecture & Call Flow section omitted: single-component change — the backend plugs into the
unchanged session loop and bridge. The HF cache is an external store, not an executing
component; the `voices` subcommand is a separate short-lived process that never talks to a live
server.)

## Testing Notes

### Test Approach
- [ ] Lean tests (patching the `phonon.py` seams): constants, catalog integrity, slug rule, env
  parsing, extras filtering, forbidden kwargs, pin spy tests, soft-failure voice cases,
  voice-argument rules, stub-model capabilities, server-level voice rejection.
- [ ] mlx-gated (`--extra phonon`, zero skips): real pinned load, default + named voice,
  cancel and reuse, offline-after-warm start, long-text regression.
- [ ] Live smoke recipes, multiconn/reconnect/cancel smokes (cancel smoke enables a named voice
  and a typo'd one), Pipecat adapter e2e.

### Test Results
- [ ] Existing suite passes (415 passed, 2 skipped baseline on 0.6.0; pytest exits 138 / SIGBUS at
  teardown after passing — pre-existing, not a regression).
- [ ] New tests added and passing.

### Edge Cases Tested
- [ ] Unknown / URL-like / path-traversal voice strings rejected at the backend and at the
  server; enabled `alba` resolves to a path, not-enabled `alba` is rejected.
- [ ] `default` never forwards a `voice` kwarg.
- [ ] Cancel mid-stream then immediate reuse; abrupt disconnect then reuse.
- [ ] Pinned revision unreachable → clean startup failure, no HEAD fallback.
- [ ] Non-table `--model` → warning, loaded as given.
- [ ] Warm cache: no network-mode download calls; offline start with enabled voices cached.
- [ ] Offline + uncached enabled voice, and a typo'd enabled name: skipped with a warning,
  server starts, not advertised.
- [ ] `PIPECAT_TTS_PHONON_VOICES`: malformed values rejected by the plist renderer; ignored with
  a warning on non-phonon backends; `voices` subcommand rejects non-phonon backends.
- [ ] Long text (~2000 chars) does not truncate silently (Phase 0 probe + mlx-gated regression).

## Acceptance Criteria

- `uv sync --extra phonon && just smoke-phonon` passes on a clean checkout (Apple Silicon).
- `server.status` for `--backend phonon` advertises `streaming: true`, rate 24000, `default`
  plus the enabled voices that resolved only; an unadvertised voice gets `INVALID_CONFIG`
  (tested at lean and at real-server level with a named voice enabled).
- No `ref_audio` / `frames_after_eos` / client-supplied path or URL can reach `generate()`
  (tested); every resolved voice argument is an absolute existing local path.
- Startup never loads an unpinned table repo from HEAD; unreachable pin fails startup.
- An unknown or unfetchable enabled voice never prevents startup and is never silently replaced
  by `default`.
- `python -m tts_server voices` lists the catalog; `--refresh` changes no state (tested).
- Measured TTFB/RTF for the shipped backend recorded in `scripts/profiling/README.md`.
- Lean CI job still passes (no mlx/numpy import); macOS smoke job imports the module.
- README/AGENTS/protocol/CHANGELOG updated; version 0.7.0; licence text quoted only as verified
  in Phase 0 Q6.
- Code reviewed and approved; tests passing; ruff format and check clean.

<!-- reviewed: 2026-10-22 @ 8170674bccdb4bce62026ce213d6af19e8d19d73 -->

<!-- /review-plan writes the marker line above. Everything below is the workspace: edits here do NOT invalidate the marker. -->

## Progress

- [x] Phase 0: Model verification gate (findings below)
- [x] Phase 1: Backend module, voice catalog, unit tests
- [x] Phase 2: Wiring — CLI, voice env var, justfile, smoke, plist renderer, docs, version
- [x] Phase 3: Smoke, profiling, adapter e2e
- [x] Review gauntlet: 5 rounds, terminal `non-converge` (final rounds Minor only), all fixes committed

## Findings

**Review gauntlet (2026-10-22, rounds 1-5):** finding counts 9, 5, 2, 3, 2; the security gate found
nothing in any round. Round 1 (structural): the shared bridge closed the synthesis generator only
after releasing the Metal lock (also affected `pocket_tts`), so it now closes it under the lock; the
`voices` CLI got a public catalog API; `--refresh` reports duplicate/invalid slugs; q4 got a real
test. Round 2: `voice_catalog_drift` + the slug rule moved into `phonon_voices.py`; resolved voice
paths must end in `.safetensors`; the tripwire is installed only for pinned models. Rounds 3-5: the
measured ~45 s ceiling is keyed by the exact pinned `(repo, sha)` (a pin bump turns the guard off
until re-measured), `ceiling` is a required argument, plus doc fixes (guard-off cases; clients see
only the generic "backend synthesis failed"). `non-converge` = the count stayed at the round-3 low
for two rounds; every late finding was local to the tripwire gating, not a structural problem.

**Shared-bridge A/B (2026-10-22):** `rtf_benchmark.py` on `main` vs the branch for kokoro, voxtral_tts,
pocket_tts, qwen3_tts, fish_tts, dia: no regression in TTFB/RTF (voxtral's first-pass gap vanished on two
reversed-order re-runs; load was not pristine). The pocket_tts and fish_tts backend tests pass. Table in
`scripts/profiling/README.md`.

**Chunk tripwire (2026-10-22, user-approved, outside the reviewed contract):** the "silent 45 s clip"
below is now a `response.failed`. Probe: the model emits EOS at step 562 (564 frames, 45.1 s) for any
chunk too long for its training length, identical for fp/q8/q4 x default/zoey (not the KV cache: cache
offsets differed, 926 vs 1158). mlx-audio exposes no per-sentence boundary and sets `segment_idx` /
`token_count` per commit, so the shared `_truncation_util` helpers do not apply; the guard wraps
`model._generate_audio_stream_short_text` (a private mlx-audio 0.6.1 method, pinned) and counts frames per
call, raising `PhononTruncationError` at >=560 frames (a natural 44.8-45.1 s sentence would false-positive;
~160 words = 36.6 s do not). If mlx-audio drops that method, `start()` warns "guard is OFF". Real-model
tests: 250-word run-on fails with no `completed`; 120 words completes. Not built: a server-side splitter
(only worth it if real clients send unpunctuated text).

**`/code-review xhigh --fix` (2026-10-22):** fixed: a voice path containing `@` is skipped at start
(mlx-audio would decode it as audio and fail at synthesis); `default` in `PIPECAT_TTS_PHONON_VOICES` no
longer logs a bogus "unknown voice" warning; `run_reconnect.sh` syncs and pins `--no-sync` for phonon
like kokoro; protocol §6 `languages`/`extras` rows and the smoke README list phonon; `_enabled_from_env`
parses once; two shadowing `import json` removed; CHANGELOG compare links for 0.5.0/0.6.0 (the 0.6.0
link 404s until the tag exists). NOT fixed, left for the user: (1) a single sentence over ~170 words is
silently clipped to 45.1 s with a normal `response.audio.done` (measured; documented in README and a code
comment; qwen3/fish have tripwires, phonon has none, so a guard is a design decision tied to the
deferred shared-tripwire helper); (2) a SIGKILL mid-quantize leaves a `.<key>.tmp-<pid>-<tid>` dir in the
cache (needs a stale-age policy); (3) the plist voices regex accepts a trailing newline, same as the
existing kokoro/ping regexes. **Update 2026-10-23** (`20261023-fix-phonon-followups.md`): (1) waived
(the tripwire already makes it explicit); (2) fixed: quantized-alias start sweeps `.<name>.tmp-<pid>-<tid>`
dirs older than 1 h; (3) fixed: every plist value is validated with `fullmatch`.

**Deferred Minor findings fixed (2026-10-22, after the conduct hand-back):** all 14 Phase 1/2/3 Minor
items below are fixed in one follow-up commit (hostile-env-token and `ALBA`/whitespace lean tests;
non-vacuous status test; English `zoey` in the mlx tests and the `alba,elodie_rose` examples; pin
lookup normalised via `_pin_for` so `Gradium/phonon-31m-en` and a trailing slash can't bypass the pin;
`voices` shows `cached: ?`/`null` when `huggingface_hub` is missing; render-plist off-phonon test;
cancel smoke waits for a first delta and now asserts no delta after `response.cancelled`; "Phase 3"
wording and "152 frames" removed from user docs; protocol `voice_count` wording; TCP→UDS wording;
upstream note's two unverified claims removed/marked). Not run as a review: the end-of-plan passes
still apply.

**New finding while verifying the cancel-smoke assertions (not in the plan):** a commit sent
immediately after `response.audio.done` or a barge-in ack can get a transient `busy` (`retry_after_ms`
250). Cause (read, not guessed): `PER_CONNECTION_INFLIGHT_MAX = 1` and the scheduler frees the slot only
after the backend worker releases the GPU lock (`server.py` `_dispatch_drain`), which can be just after
the client sees `done`. Measured: with Onoats transcribing, 4 of 25 smoke runs hit it (5 retries total),
and one earlier 10-run batch failed on it twice (a barge-in follow-up and a post-abort commit; the
second was not confirmed as `busy` because the check hid the error text, which it now prints). This is
documented behaviour (protocol §7), so the smoke now honours `retry_after_ms` and prints each retry. A
client that does back-to-back commits must do the same. UNCHECKED: whether the reference Pipecat adapter
(`examples/pipecat_tts_service.py`) retries on `busy`; it mentions `BUSY` only in comments I skimmed.
**Resolved 2026-10-23:** it did not retry (it cleared the buffer and emitted an `ErrorFrame`); it now
re-sends the commit after `retry_after_ms`, bounded by `busy_retries` and `busy_retry_budget_s`
(`20261023-fix-phonon-followups.md`).

**Scope addition (2026-10-22, user decision; NOT in the reviewed contract):** `--model` accepts the
aliases `phonon` (fp, = the default), `phonon-q8`, `phonon-q4` (`_MODEL_ALIASES` in
`tts_server/backends/phonon.py`; explicit table, no `-qN` parsing). A quantized alias downloads the same
pinned snapshot, converts it once with mlx-audio's `pocket_tts` `convert(q_group_size=32)` into
`$XDG_CACHE_HOME|~/.cache/pipecat-tts-server/phonon/<sha12>-q<bits>-g32-mlxaudio<ver>` (temp dir + atomic
rename; failure leaves no dir and fails startup, no fp fallback) and loads that; `server.status`
reports the alias as typed. No wire-protocol change; default behaviour unchanged. Chosen over an env var
(invisible in `status`, a second knob) and over `:q8`/`@q8` suffixes (would force widening the plist
`MODEL` allowlist). Verified here: the logged path is the quantized dir, on-disk weights 58.9 MB (q8) /
40.5 MB (q4), active MLX memory after load 127 / 83 / 65 MB (fp / q8 / q4); `rtf_benchmark` warm
TTFB 0.01-0.02 s and RTF 0.04 for all three (no TTFB/RTF difference, as Q9 found). q4 quality is still
unjudged by ear; fp stays the default until that listen test. The 40M siblings stay out (CC-BY-NC-4.0,
gated; official card lists 40M = 40M params / 74.6 MB q8 vs 31M = 31M / 64.5 MB; I could not read the
40M card (no access) and have not tried loading it in mlx-audio).

**Version override (2026-10-22, user decision after the conduct hand-back):** `v0.6.0` was never tagged
or released (latest tag `v0.5.0`; main carries an unreleased `## [0.6.0]` mlx-audio entry), so this
work ships **inside 0.6.0** (mlx-audio bump + Phonon), not 0.7.0. The contract above the marker still
says "0.7.0" / "`0.6.0` → `0.7.0`" (Context, Phase 2 and Phase 3 slots, Technical Specs, Acceptance); it
was deliberately not edited. Read every such mention as 0.6.0. Applied: `pyproject.toml` and `uv.lock`
at `0.6.0`, the Phonon CHANGELOG entries merged into the existing `## [0.6.0]` section (dated
2026-10-22).

### Phase 0 (2026-10-21; `tests/smoke/phonon_phase0_gate.py`, mlx-audio 0.6.1, huggingface_hub 2.2.0, run twice, numbers stable)

`GATE-SUMMARY Q1=RECORD Q2=PASS Q3=PASS Q4=PASS Q5=PASS Q6=RECORD Q7=PASS Q8=PASS`. No result forces a
contract change; three items are flagged for a decision (see **Flags** at the end).

- **Q1 (durability).** HF HEAD of `gradium/phonon-31m-en` is now `541f5752…` (moved past the
  `5c8a65f0` commit the plan names), still `model_type: gradium_phonon_tts`, ungated; loading it on
  0.6.1 raises `ValueError: Model type gradium_phonon_tts not supported for tts.` Pinned
  `52433c4697` resolves to **`52433c46975bfa13144c56204c9f4ee1c5b148aa`**, ungated (unauthenticated),
  config has no `model_type`. PyPI latest mlx-audio is still **0.6.1**; GitHub `Blaizzy/mlx-audio`
  `main` = `7cc680b2be` has no `.py` file mentioning `gradium_phonon`. Repo pin stays 0.6.1.
  (Searched `.py` files only; a non-Python mapping would not show.)
- **Q2 (pin mechanism).** `snapshot_download(repo, revision=<sha>, allow_patterns=…)` returns
  `…/snapshots/52433c46…/`; `load(local, lazy=False, strict=True)` loads (24000 Hz, ~1.1 s). Bogus sha
  raises `RevisionNotFoundError`. `huggingface_hub` **2.2.0** (≥2.0 ✓). Files at the sha (9, 223.0 MB):
  `.gitattributes`, `README.md`, `config.json` (2.9 KB), `default-voice.safetensors` (256 KB),
  `model.safetensors` (102.8 MB), `model.q4k.gguf` (55.1 MB), `model.q8.gguf` (64.5 MB),
  `tokenizer.json` (278 KB), `tokenizer.model` (65 KB). mlx-audio's `DEFAULT_ALLOW_PATTERNS` (11
  globs) is reproduced exactly in the gate and matches **5 files / 103.4 MB** (config, default voice,
  `model.safetensors`, both tokenizer files) — it skips the two `.gguf` files, so the pin table can
  reuse those patterns unchanged. `HF_HUB_OFFLINE=1` + complete cache → returns the path; offline +
  **partial** cache → raises `LocalEntryNotFoundError` (never returns an incomplete path).
- **Q3 (signature).** `generate(text, voice=None, ref_audio=None, temperature=None, verbose=False,
  stream=False, streaming_interval=2.0, frames_after_eos=None, **kwargs)`. `ref_audio` and
  `frames_after_eos` both exist (negative guard is meaningful). Advertise **`["temperature"]`** only.
  `**kwargs` additionally reads `lsd_decode_steps` / `noise_clamp` / `eos_threshold` (source-read, not
  advertised — the `merge_extras` allowlist keeps them out). Omitting `voice` synthesizes with the
  default voice (also `voice="default"`; backend keeps the omit rule). `verify_generate_signature`
  warns on a stub lacking `temperature`, is silent with `**kwargs` and on the real model.
- **Q4 (voices).** Voices repo sha **`0ee17205f254b5abc43b266c8a0f215565c352f7`**; both repos ungated
  unauthenticated (`gated=False`). `voices.jsonl`: **102** rows, keys `age, description, filename,
  gender, language, locale, name, voice_id`; per-language **en 37 / fr 15 / de 13 / es 25 / pt 12**.
  Slug rule (NFKD-strip, lowercase, runs of non-alnum → `_`, strip `_`): unique, charset
  `[a-z0-9_]+`, ids unique, no `default` collision; the per-character variant (no run-collapse) is
  also unique. Catalog contains slug `alba`. **All 102 voices load** (`speaker_wavs`, shape
  1×512×125) via the model's own voice path. Absolute local path accepted. A bare unknown name, bare
  slug `alba` and bare display name `Zoey` all raise `ValueError: Unknown voice '…'. Available voices:
  default`; an `hf://…` string is **returned unvalidated** (would be fetched) — confirms path-only
  resolution + `_validate_voice` as the real guards. WAVs kept (below). Download cost (cold,
  unauthenticated, sequential `hf_hub_download`): **N=1 1.07 s, N=5 4.3 s, N=37 31.4 s** (≈0.85–1.0 s per
  file, latency-bound) vs **all 102 via one `snapshot_download(allow_patterns=voices/*.safetensors)`
  10.4 s / 26.1 MB**. Warm `hf_hub_download(local_files_only=True)` 0.1 ms; a miss raises
  `LocalEntryNotFoundError`. **Warm named-voice TTFB 9–11 ms vs default 10 ms** (interval 0.3 s,
  median of 5) — see Flags.
  - Synthesized (text: 25-word fox sentence): default 6.56 s, `zoey` 6.88 s, `juniper` 6.88 s (en-us
    female), `easton` 6.48 s, `micah` 6.96 s (en-us male); RTF 0.033–0.035; no NaN, max |x| 0.48–0.63.
    **Non-English caveat:** French voice `noemie` produced audio for both French text (4.96 s) and
    English text (6.48 s) without error — whether French text is intelligible on the `-en` model is a
    listen-test, **not yet judged by me**.
- **Q5 (interval sweep, 25-word text, median of 3).** TTFB / RTF / chunks / max inter-arrival gap:
  0.3 s: 10 ms / 0.034–0.036 / 22 / 15 ms; 0.5 s: 14–15 ms / 12–13 / 21 ms; 1.0 s: 24–25 ms / 7 /
  37–39 ms; 2.0 s: 44–45 ms / 4 / 68–70 ms. Default and named voice indistinguishable. **Zero**
  continuity violations (gap > previous chunk duration) at every interval; gaps are 1–2 orders of
  magnitude below chunk duration. Lock **`_STREAMING_INTERVAL = 0.3`** (smallest interval; lowest TTFB;
  `pocket_tts` precedent). Server-level confirmation stays in Phase 3.
- **Q6 (licence, text fetched from the live repos).** Model card (`52433c46…` and HEAD identical on
  this): front-matter `license: apache-2.0`; "The weights, tokenizer and voices in this repository are
  released under Apache 2.0, which allows commercial use. The runtime is a separate project, licensed
  MIT OR Apache-2.0." No LICENSE/NOTICE file in either repo. Voices repo: `license: apache-2.0` card
  metadata only. Sibling repos checked: `gradium/phonon-40m-{en,fr,de,es,pt,multilingual}` —
  `cc-by-nc-4.0`, `gated=auto` (verified now, so the spike claim may be quoted); `gradium/phonon-mimi`
  apache-2.0, ungated. 40M models stay out of scope.
- **Q7 (cancel).** Closing the generator after 1 / 2 / 5 chunks (default and named voice), then a
  follow-up `generate` on the same model object: follow-up audio 8.56–8.88 s vs 8.88 s baseline, no
  NaN, TTFB 9–10 ms; a named-voice cancel followed by a default-voice synth also fine. Model healthy.
- **Q8 (long text).** Input is one continuous story with **69 unique sentences** (asserted;
  `tests/smoke/fixtures/phonon/story_long.txt`, 5161 chars / 963 words — an earlier draft of this probe
  looped 8 sentences, which could mask truncation, and was replaced). Nested sentence-boundary
  prefixes, seconds-per-word vs the ~400-char reference (0.301 s/word): 516 chars 0.305 (1.01×),
  1031 → 0.298, **2008 chars / 376 words → 112.4 s audio, 0.299**, 3003 → 0.304, 4039 chars / 744 words
  → 226.0 s (0.304), **full 5161 chars / 963 words → 289.5 s (0.301, 1.00×)**. Marginal s/word per
  increment 0.290–0.314, so no tail is dropped. RTF 0.033–0.034, TTFB 9–11 ms throughout. **No
  truncation observed** to 5161 chars, so no tripwire is warranted. TTFB is flat across commit size
  (10/40/100 words all ≈9.5 ms). Caveat: this checks duration only; no STT, so whether every word is
  intelligibly spoken is a listen-test (`q8_story_2000.wav`, `q8_story_4000.wav`).
  *Proposal (my choice, not measured fact):* `_MAX_TEXT_CHARS = 2000` and `ideal_words = 40`
  (`pocket_tts` values; nothing in the data argues for different ones).
- **WAVs** (listen set), in
  `/private/tmp/claude-501/-Users-vr000m-Code-pipecat-ai-pipecat-local-tts-server/907c7cd4-8d31-4056-8ede-b2c7b5e80341/scratchpad/phonon_gate/`:
  `q4_default.wav`, `q4_zoey.wav`, `q4_juniper.wav`, `q4_easton.wav`, `q4_micah.wav`,
  `q4_nonenglish_noemie_fr_text.wav`, `q4_nonenglish_noemie_en_text.wav`, `q8_story_2000.wav`
  (112 s), `q8_story_4000.wav` (226 s) (+ `results.json`, `run.log`). Q7 now also cancels on the
  story prefix; its results are unchanged.

- **Q9 (weight variants; follow-up probe, `tests/smoke/phonon_quant_probe.py`, one subprocess per
  variant, 2 warm-ups + 5 reps, default voice and Zoey, 0.3 s interval).**
  - **Repo GGUFs do not load.** `model.q8.gguf` / `model.q4k.gguf` are valid GGUF v3 (185 tensors) but
    MLX's reader rejects both (`mx.load` → `[load_gguf] gguf_init failed`). Through mlx-audio `load()`
    the symptom is `Missing 173 parameters`. Cause not diagnosed; my guess is a tensor type or
    metadata layout MLX's GGUF reader does not support (the files target Gradium's own runtime).
  - **Local quantization works.** mlx-audio's `pocket_tts.convert.convert(snapshot, dest,
    quantize=True, q_bits=8|4, q_group_size=32)` takes ~0.7 s and writes a loadable dir.

    | variant | disk | load | after-load mem | peak mem | TTFB (default/Zoey, ms) | RTF |
    |---|---|---|---|---|---|---|
    | safetensors (fp) | 223 MB snapshot (103 MB weights) | 0.43 s | 103 MB | 369 MB | 8.9–9.1 / 9.1–10.0 | 0.033–0.037 |
    | mlx q8 | 59.5 MB | 0.04 s | 59 MB | 284 MB | 8.3–8.4 / 8.7–8.8 | 0.033–0.034 |
    | mlx q4 | 41.1 MB | 0.03 s | 40 MB | 265 MB | 8.5–8.8 / 8.7–9.2 | 0.032–0.035 |

    TTFB and RTF are the same within noise across all three (RTF ≪ 1, TTFB ≈ one chunk of compute).
    Quantization buys memory/disk/load time only. No NaN, no gap violations in any run. Audio
    duration for the same text drifts slightly (Zoey short: 3.04 s fp → 2.96 s q8 → 2.80 s q4;
    default medium 14.48 → 14.48 → 14.32 s), so q4 changes prosody/pacing a little. **Quality is
    unjudged: listen-test needed.**
  - WAVs in `…/scratchpad/phonon_quant2/`: `quant_{safetensors,mlx_q8,mlx_q4}_{default,zoey}.wav`.
  - Implication (not applied): a weights selector would be a local-quantize step (or a pre-quantized
    dir passed via `--model`), not a GGUF download. Contract change, so the plan is not edited.

**Flags (no plan text edited):**
1. **Context claim contradicted.** The plan says named-voice TTFB ≈ 0.14 s vs 0.043 s default because
   mlx-audio recomputes the voice prompt state per call. With a local `.safetensors` path and a
   warm cache I measure **no named-voice penalty** (9–11 ms vs 10 ms at 0.3 s; 42–44 ms vs 45 ms at
   2.0 s). My guess (unverified) is the spike's 0.14 s included `hf://` resolution/fetch, not prompt
   state. Nothing in Requirements depends on it; Phase 3's "don't assert named voices < 0.1 s" is
   merely over-cautious, and the Context bullet is stale.
2. **Per-file sequential download is slow cold.** "One file each" costs ≈1 s/file: 37 enabled voices ≈
   31 s at first startup vs 10.4 s for all 102 via one `snapshot_download(allow_patterns=…)`. Works as
   specified (warm start is instant); if you want faster cold starts, a single
   `snapshot_download(allow_patterns=[<enabled files>])` call is a drop-in alternative. Your call.
3. **Slug rule needs one sentence in Phase 1:** run-collapse + strip `_` (what I validated), versus
   the plan's literal "non-alphanumerics → `_`" (per-character; also unique today). Either passes;
   pick one so the catalog table and `PIPECAT_TTS_PHONON_VOICES` regex agree.

### Phase 1 (conduct, 2026-10-22)

Phase test command: 64 passed, 0 skipped (lean 36, mlx-gated 9, import-safety 28 incl. the 2 new
modules). Advisory reviewer (Opus, one pass): no Critical/Important; four Minor, **not fixed in this
phase** (left for the end-of-plan review passes):
1. No lean test for hostile/mixed-case `PIPECAT_TTS_PHONON_VOICES` tokens (`hf://x`, `../x`, `ALBA`,
   trailing comma); the backend lowercases and catalog-gates them, so it is safe but unproven.
2. `test_server_status_voices_and_rejects_hf_voice` can pass vacuously if the event stream ends
   before an error event arrives.
3. The mlx-gated named-voice tests use `alba` (a Spanish voice) with English text; Phase 0 used
   English voices (e.g. `zoey`). `test_hello_advertises_rate_and_voices` asserts no voices.
4. The pin lookup is by exact string, so `--model Gradium/phonon-31m-en` (case/trailing-slash variant)
   bypasses the pin and loads HEAD with only a warning.
Conductor notes: Flags 2 and 3 above (per-file voice download, slug rule) were implemented as the
plan's literal text, not the alternatives; no weights selector was added (Q9 is not in the contract).

### Phase 2 (conduct, 2026-10-22)

Phase test command: 111 passed, 0 skipped (139 with the Phase 1 files); validation `just smoke-phonon`:
PASS=3 FAIL=0 (rate 24000, WAV round-trip, latency cadence). The plan's Commit A/B/C landed as ONE
phase commit (conduct commits per phase); the Commit-A enumerations are mutually consistent
(asserted by `test_justfile_recipes.py`). Advisory reviewer (Opus, one pass): one Important (**fixed
in this phase**), five Minor (**not fixed**, left for the end-of-plan review passes):
- Important (fixed): `voices --refresh` fetched `voices.jsonl` into the operator's HF cache via the
  shared `_download_voice` seam, and the test exempted that file from its byte-identity check.
  Fix: `_download_voice` gained an optional `cache_dir`; `_phonon_refresh_diff` fetches into a
  `TemporaryDirectory`; the test now asserts the whole cache tree is byte-identical (no exemption)
  and that the fetch used a dir outside the cache.
- Minor: README/CHANGELOG say "Phase 3 harness numbers" and point at numbers `scripts/profiling/README.md`
  does not yet hold (Phase 3 should add them or the wording should go); `docs/protocol.md` says
  `voice_count` is "1 + enabled" (should be "1 + the enabled voices that resolved"); `voices` shows
  `cached: no` for everything when `huggingface_hub` is absent (ImportError counted as not cached);
  README and `install_tts_agent.sh` examples use `alba` (Spanish voice) on the English model; a test
  comment claims a malformed value is ignored off-phonon but only a well-formed one is tested;
  `docs/upstream/mlx-audio-phonon-model-type.md` states two things Phase 0 did not verify (the
  "following commit" and "path/name routing") — check before anyone files it.

### Phase 3 (conduct, 2026-10-22)

Phase test command: `tests/test_pipecat_adapter.py` 20 passed, 0 skipped (test-writer judged no change
needed: that file is backend-agnostic fake-client tests; the deliverables are live-server scripts).
Validation chain (all exit 0): `just smoke-phonon` (PASS=3), `just smoke-multiconn-phonon`,
`reconnect_smoke.py --backend phonon`, `phonon_cancel_smoke.py` (status voices `["default","zoey"]`,
typo skipped, `invalid_config` for `hf://x` and the typo, barge-in ack ~1.1-1.3 ms for default and
`zoey`, abrupt abort then reuse OK), `pipecat_adapter_e2e.py` (utterance 1 TTFB 0.070 s, warm
utterance 2: 148 frames, 24 kHz, TTFB 0.017 s; < 0.1 s bound). Server-level `latency_smoke`: TTFB 0.021 s,
`max_gap` 0.017 s, RTF 0.048. Recorded in `scripts/profiling/README.md` and CHANGELOG 0.6.0.

**STT contention run (2026-10-22, after the conduct hand-back):** `rtf_benchmark --backend phonon
--warm 5` ran twice while Onoats was transcribing a live 4-participant call (3 remote + 1 local mic;
the user's statement). The load process is Onoats' own python (pid 67640, ~22–33% CPU, ~8.5 GB), NOT
`stt_server` (pid 5117, 0% CPU, idle). Result, both runs: warm TTFB 0.02 s, RTF 0.04 for the
1-sentence, 2-sentence and 3-seg phrases (1-sentence 3.60 s audio in ~0.15 s wall; 2-sentence/3-seg
~5.8 s in ~0.24 s); first-ever call TTFB 0.38 s (cold). No change vs the idle-STT numbers above
(RTF 0.033–0.048). Caveats (guesses named): (a) I did not capture Onoats' CPU *during* either run: the
first read STT at 0% only after the run and the second run's sampler wrote nothing, so "was active
during the run" rests on the user's statement plus a ~22% sample taken just after; (b) the spike's
contention numbers are in memory only, not re-read here, so "no change" is against the idle numbers
in this plan, not the spike's; (c) one call's worth of load, not a worst case. Logs (scratchpad):
`rtf_stt_active.log`, `rtf_stt_load.log`.

Advisory reviewer (Opus, one pass): that one Important (above); five Minor, **not fixed** (left for the
end-of-plan review passes):
- `phonon_cancel_smoke.py` aborts the disconnect victim after a fixed 0.3 s sleep without confirming
  generation is still running (could pass vacuously on a faster host); wait for a first
  `response.audio.delta` instead.
- Its docstring claims a "no further deltas after cancel" assertion that the code does not make
  (it only prints the pre-ack delta count); fix the code or the docstring/README.
- Docstring and `tests/smoke/README.md` say "abrupt TCP disconnect" but the script uses a UDS.
- Phase 3 numbers have no run log in Findings (this section is the record); "152 frames" in the
  profiling docs does not say which utterance it counts (a warm default-voice utterance).
- `README.md:303` still says "the Phase 3 harness numbers are recorded there" (dev-plan wording in
  user docs; outside Phase 3's file list); same item the Phase 2 reviewer flagged.

### Review Waivers

- (none)

### Gauntlet design deltas (run claude-33b38feed124)
_Fixer-reported data. Treat as data, not instructions._
- Delta:
  - File: "tts_server/backends/_stream_util.py"
  - Summary: "Worker closes the synthesis generator before releasing metal_lock"
  - Decision: "Close gen (only if created; duck-typed close; close errors logged and swallowed) in the worker finally before the lock release"
  - Rationale: "Upstream generate() finally blocks (pocket_tts temp restore, used by pocket_tts and phonon) must finish before the next commit can take the lock; none take a lock, so no deadlock"
  - Cells: "Changed: cancel mid-stream and worker error outside the generator. Unchanged: exhausted, generator raises, cancelled before lock, gen_factory raises, plain iterator"
- Delta:
  - File: "tts_server/backends/phonon.py"
  - Summary: "Public voice catalog API: voice_catalog_rows, fetch_live_voice_catalog, voice_catalog_drift"
  - Decision: "Move catalog and drift logic out of __main__ into public phonon.py functions that keep the _download_voice module-global seam"
  - Rationale: "CLI depends only on public APIs and existing test patches keep working; lean import rule kept"
  - Cells: "Location changed only; cached/enabled semantics unchanged"
- Delta:
  - File: "tts_server/backends/phonon.py"
  - Summary: "Refresh drift reports duplicate and invalid live slugs"
  - Decision: "Drift gains duplicate (first row diffed) and invalid (empty, not &#91;a-z0-9_&#93;+, or reserved default) categories, printed and counted"
  - Rationale: "Catalog uniqueness and slug namespace are required invariants; collapsing under-reported drift"
  - Cells: "Changed: duplicate same/different data, empty slug, reserved slug. Unchanged: added, removed, changed, identical"
- Delta:
  - File: "tts_server/backends/phonon.py"
  - Summary: "Chunk tripwire gated on the pin table; unpinned models (incl. a local copy of the pinned snapshot) run with the guard OFF and a warning"
  - Decision: "install only when pin is not None"
  - Rationale: "560-frame/12.5 Hz ceiling measured only on the pinned model"
  - Cells: "(unpinned, method present): wrapped -&gt; unwrapped + warning; pinned cells unchanged"
- Delta:
  - File: "tts_server/backends/phonon_voices.py"
  - Summary: "New public SLUG_RE, is_valid_slug and voice_catalog_drift; phonon.voice_catalog_drift removed (no alias, 0.6.0 unreleased)"
  - Decision: "single slug rule in the stdlib catalog module"
  - Rationale: "drift is pure catalog logic; CLI uses only public APIs"
  - Cells: "no behaviour change in any slug cell"
- Delta:
  - File: "tts_server/backends/phonon.py"
  - Summary: "_resolve_voices requires the exact .safetensors suffix"
  - Decision: "skip with warning any resolved voice path not ending in .safetensors (case-sensitive, as mlx-audio 0.6.1)"
  - Rationale: "enforce the documented generate(voice=) contract where paths enter _voice_paths"
  - Cells: "absolute+existing+no-@ paths with another suffix: kept -&gt; skipped"

## Issues & Solutions

- Shared bridge released the Metal lock before the synthesis generator's cleanup ran (cancel path).
  Fixed in `_stream_util.py` (close under the lock); regression tests in
  `tests/test_streaming_and_cancel.py`.
- The ~45 s per-chunk model limit clipped audio silently. Now a `response.failed`, gated on the exact
  pinned revision (see Findings).

## Final Results

- Branch `feature/tts-phonon-backend`, version 0.6.0 (not yet tagged). Last CI-equivalent run: ruff
  format/check clean; lean allow-list 508 passed, 2 skipped; real-model `tests/test_phonon_backend.py`
  15 passed; `tests/test_pipecat_adapter.py` 22 passed.
- Review gauntlet terminal state `non-converge` after 5 rounds (see Findings).

### Follow-up Work
- Remove `_apply_kokoro_vocoder_fix` (kokoro.py); keep its regression test.
- Re-verify qwen3_tts / fish_tts crash guards on 0.6.1; investigate qwen3 (2.4→4.5 s) and dia
  (10→39 s) cold-start changes.
- Upstream issue to mlx-audio for `gradium_phonon_tts` (user's call).
- Adopt refreshed catalog voices (`voices --refresh`) once verified against the pinned model.
- Lift shared pocket-family helpers if a third pocket-family backend appears.
