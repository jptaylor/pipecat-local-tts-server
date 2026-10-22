#!/usr/bin/env python3
"""Gradium Phonon (``phonon``) Phase 0 verification gate (dia/qwen3/fish precedent —
run BEFORE any backend code).

Standalone script: direct mlx-audio + huggingface_hub usage, no tts_server server
involvement. Answers Q1-Q8 of the plan
(``docs/dev_plans/20261021-feature-tts-phonon-backend.md``), printing a labelled
PASS/FAIL/RECORD line per question plus a machine-greppable final summary
(``GATE-SUMMARY ...`` / ``phonon phase0 gate: ...``). Re-runnable: Gradium may
re-upload again, so every live-repo fact is re-fetched, never hard-coded (the
only constant is the *short* pinned revision, resolved to a full sha in Q2).

  Q1 — revision-durability detector: does HF HEAD still fail with
       ``gradium_phonon_tts``? is the pinned sha fetchable + ungated? has a newer
       mlx-audio (PyPI wheel / GitHub main) added support? (informational)
  Q2 — pin mechanism: ``snapshot_download(revision=<40-char sha>)`` + local load;
       bogus sha raises; huggingface_hub >= 2.0; file list/size at the sha;
       offline behaviour on a complete vs partial snapshot.
  Q3 — ``generate()`` signature, extras, forbidden kwargs, voice=None omit rule,
       ``verify_generate_signature`` drift warning.
  Q4 — voices: catalog from ``voices.jsonl`` at a recorded sha, slug uniqueness,
       unauthenticated gating, load EVERY voice, synthesize named voices (WAVs
       kept), absolute-path acceptance, bare-name behaviour, download cost vs N,
       named-voice TTFB, one non-English caveat.
  Q5 — ``streaming_interval`` sweep (TTFB / RTF / max inter-chunk arrival gap).
  Q6 — licence/gating text for model + voices repos (and any sibling repos checked).
  Q7 — cancel at chunk boundaries (default + named voice) then reuse the model.
  Q8 — long-text probe: duration scales with word count? ceiling for
       ``_MAX_TEXT_CHARS``.

Run: ``uv run --extra pocket_tts python tests/smoke/phonon_phase0_gate.py``
(``--extra pocket_tts`` because the ``phonon`` extra does not exist until Phase 1).
WAVs and ``results.json`` land in ``--out-dir`` (default: fresh secure mkdtemp).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

from _gate_helpers import _audio_sanity, _to_numpy, _write_wav

MODEL_REPO = "gradium/phonon-31m-en"
VOICES_REPO = "gradium/phonon-voices"
PINNED_SHORT = "52433c4697"
ALLOW_PATTERNS = [  # mirrors mlx_audio.utils.DEFAULT_ALLOW_PATTERNS (verified in Q2)
    "*.json",
    "*.safetensors",
    "*.py",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.jinja",
    "*.jsonl",
    "*.yaml",
    "*.npz",
    "*.pth",
]
# Candidate Phase-1 constant under test (pocket_tts precedent); Q5 decides.
CANDIDATE_INTERVAL = 0.3

TEXT_SHORT = (
    "The quick brown fox jumps over the lazy dog, and then it quietly walks "
    "back home before the evening rain begins to fall."
)  # 25 words
TEXT_MULTI = (
    "Good morning. This is a short test of the voice. It has a few sentences, "
    "so the model needs to split it. Each sentence should come out clearly."
)

# One continuous story, every sentence unique (a looped text could mask truncation or make
# the model's behaviour unrepresentative); see fixtures/phonon/README.md. Q7/Q8 use nested
# prefixes of it cut at sentence boundaries.
_STORY_PATH = Path(__file__).parent / "fixtures" / "phonon" / "story_long.txt"
_STORY_PARAGRAPHS = [
    " ".join(p.split()) for p in _STORY_PATH.read_text().split("\n\n") if p.strip()
]
_STORY_SENTENCES = [
    s for p in _STORY_PARAGRAPHS for s in re.split(r"(?<=[.!?])\s+", p.strip()) if s
]


def _story_prefix(n_chars: int) -> str:
    """Nested prefix of the story: whole sentences until >= n_chars (or the full story)."""
    out = ""
    for s in _STORY_SENTENCES:
        if len(out) >= n_chars:
            break
        out += s + " "
    return out.strip()


def _tag(q: str, level: str, msg: str) -> None:
    print(f"[{q.upper()}] {level}: {msg}")


def _rec(q: str, msg: str) -> None:
    print(f"   RECORD [{q}] {msg}")


def _med(xs):
    return statistics.median(xs) if xs else float("nan")


# --------------------------------------------------------------------------- #
# generate() timing helper
# --------------------------------------------------------------------------- #
def _timed_generate(model, text, *, voice=None, interval=CANDIDATE_INTERVAL, close_after=None):
    """Drive generate(stream=True); return dict with TTFB, RTF, per-chunk arrival times,
    chunk durations and the concatenated audio. ``close_after`` closes the generator
    after that many chunks (cancel probe)."""
    kwargs = {}
    if voice is not None:
        kwargs["voice"] = voice
    t0 = time.perf_counter()
    gen = model.generate(text, stream=True, streaming_interval=interval, **kwargs)
    arrivals, durs, parts = [], [], []
    rate = int(model.sample_rate)
    try:
        for res in gen:
            arrivals.append(time.perf_counter() - t0)
            a = _to_numpy(res.audio)
            parts.append(a)
            durs.append(a.shape[0] / rate)
            if close_after is not None and len(arrivals) >= close_after:
                break
    finally:
        gen.close()
    total = time.perf_counter() - t0
    import numpy as np

    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    dur = audio.shape[0] / rate
    gaps = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]
    violations = sum(1 for i, g in enumerate(gaps) if g > durs[i])
    return {
        "ttfb": arrivals[0] if arrivals else float("nan"),
        "total": total,
        "audio_s": dur,
        "rtf": total / dur if dur else float("nan"),
        "n_chunks": len(arrivals),
        "chunk_durs": durs,
        "max_gap": max(gaps) if gaps else 0.0,
        "gap_violations": violations,
        "audio": audio,
    }


# --------------------------------------------------------------------------- #
# Voice catalog helpers
# --------------------------------------------------------------------------- #
def _slug(name: str, *, collapse: bool = True) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    if collapse:
        return re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return re.sub(r"[^a-z0-9]", "_", s)


def _fetch_catalog(api_sha: str, cache_dir=None):
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        VOICES_REPO,
        "voices.jsonl",
        revision=api_sha,
        token=False,
        cache_dir=cache_dir,
    )
    rows = [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    return rows


# --------------------------------------------------------------------------- #
# Q1
# --------------------------------------------------------------------------- #
def _http_json(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers={"User-Agent": "phonon-phase0-gate"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https URLs
        return json.loads(r.read())


def _http_bytes(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "phonon-phase0-gate"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return r.read()


def q1_durability(res: dict) -> str:
    from huggingface_hub import HfApi, hf_hub_download

    q = "q1"
    out: dict = {}
    api = HfApi()
    head = api.model_info(MODEL_REPO, token=False)
    out["head_sha"] = head.sha
    out["head_gated"] = head.gated
    cfg = Path(hf_hub_download(MODEL_REPO, "config.json", revision="main", token=False))
    head_cfg = json.loads(cfg.read_text())
    out["head_model_type"] = head_cfg.get("model_type")
    _rec(q, f"HF HEAD sha={head.sha} gated={head.gated} model_type={out['head_model_type']!r}")

    pinned = api.model_info(MODEL_REPO, revision=PINNED_SHORT, token=False)
    out["pinned_full_sha"] = pinned.sha
    out["pinned_gated"] = pinned.gated
    ok_pin = bool(pinned.sha) and len(pinned.sha) == 40 and not pinned.gated
    _rec(q, f"pinned {PINNED_SHORT} -> {pinned.sha} gated={pinned.gated} fetchable={ok_pin}")
    pcfg = json.loads(
        Path(
            hf_hub_download(MODEL_REPO, "config.json", revision=pinned.sha, token=False)
        ).read_text()
    )
    out["pinned_model_type"] = pcfg.get("model_type")
    _rec(q, f"pinned config.json model_type={out['pinned_model_type']!r}")

    # HEAD load under the installed mlx-audio 0.6.1
    from importlib.metadata import version as _pkg_version

    from mlx_audio.tts.utils import load

    out["mlx_audio_installed"] = _pkg_version("mlx-audio")
    try:
        load(MODEL_REPO, lazy=False, strict=True)  # HEAD
        out["head_loads_on_installed"] = True
    except Exception as exc:  # noqa: BLE001 - recording the failure is the point
        out["head_loads_on_installed"] = False
        out["head_load_error"] = f"{type(exc).__name__}: {exc}"[:300]
    _rec(
        q,
        f"HEAD load under mlx-audio {out['mlx_audio_installed']}: "
        f"loads={out['head_loads_on_installed']} {out.get('head_load_error', '')}",
    )

    # Newer mlx-audio? PyPI metadata + a read-only grep of the wheel / GitHub main.
    try:
        meta = _http_json("https://pypi.org/pypi/mlx-audio/json")
        latest = meta["info"]["version"]
        out["pypi_latest"] = latest
        urls = meta["info"].get("project_urls") or {}
        out["project_urls"] = urls
        _rec(q, f"PyPI latest mlx-audio={latest} project_urls={urls}")
        if latest != "0.6.1":
            wheel = next(
                (u for u in meta["urls"] if u["packagetype"] == "bdist_wheel"),
                None,
            )
            if wheel:
                data = _http_bytes(wheel["url"])
                hits = []
                with zipfile.ZipFile(io.BytesIO(data)) as z:  # read in memory, never extracted
                    for n in z.namelist():
                        if n.endswith(".py"):
                            txt = z.read(n).decode("utf-8", "replace")
                            if "gradium_phonon" in txt:
                                hits.append(n)
                out["pypi_latest_mentions_gradium_phonon"] = hits
                _rec(q, f"PyPI {latest} wheel files mentioning gradium_phonon: {hits or 'none'}")
        else:
            out["pypi_latest_mentions_gradium_phonon"] = "n/a (latest == installed 0.6.1)"
    except Exception as exc:  # noqa: BLE001
        out["pypi_error"] = f"{type(exc).__name__}: {exc}"[:200]
        _rec(q, f"PyPI check failed: {out['pypi_error']}")

    try:
        repo_url = (out.get("project_urls") or {}).get("Homepage") or (
            out.get("project_urls") or {}
        ).get("Source")
        m = re.search(r"github\.com/([^/]+/[^/#?]+)", repo_url or "")
        slug = m.group(1).removesuffix(".git") if m else None
        out["github_repo"] = slug
        if slug:
            c = _http_json(f"https://api.github.com/repos/{slug}/commits/main")
            out["github_main_sha"] = c["sha"]
            raw = _http_bytes(f"https://codeload.github.com/{slug}/tar.gz/main")
            hits = []
            with tarfile.open(
                fileobj=io.BytesIO(raw), mode="r:gz"
            ) as tf:  # streamed, no extraction
                for mem in tf:
                    if mem.isfile() and mem.name.endswith(".py") and mem.size < 2_000_000:
                        f = tf.extractfile(mem)
                        if f and b"gradium_phonon" in f.read():
                            hits.append(mem.name)
            out["github_main_mentions_gradium_phonon"] = hits
            _rec(
                q,
                f"GitHub {slug}@main={c['sha'][:10]} files mentioning gradium_phonon: {hits or 'none'}",
            )
    except Exception as exc:  # noqa: BLE001
        out["github_error"] = f"{type(exc).__name__}: {exc}"[:200]
        _rec(q, f"GitHub check failed: {out['github_error']}")

    res[q] = out
    _tag(q, "RECORD", "informational — repo pin stays 0.6.1; bump is a separate change")
    return "RECORD" if ok_pin else "FAIL"


# --------------------------------------------------------------------------- #
# Q2
# --------------------------------------------------------------------------- #
_OFFLINE_SNIPPET = r"""
import json, sys
from huggingface_hub import snapshot_download
repo, sha, cache, pats = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
try:
    p = snapshot_download(repo, revision=sha, cache_dir=cache or None, allow_patterns=pats)
    import os
    print(json.dumps({"ok": True, "path": p, "files": sorted(os.listdir(p))}))
except BaseException as e:
    print(json.dumps({"ok": False, "err": type(e).__name__ + ": " + str(e)[:200]}))
"""


def _offline_snapshot(sha, cache, patterns):
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            _OFFLINE_SNIPPET,
            MODEL_REPO,
            sha,
            cache or "",
            json.dumps(patterns),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"ok": False, "err": f"unparseable: {r.stdout[-200:]} {r.stderr[-200:]}"}


def q2_pin(res: dict):
    import huggingface_hub
    from huggingface_hub import HfApi, snapshot_download
    from mlx_audio.tts.utils import load

    q = "q2"
    out: dict = {}
    ok = True
    sha = res["q1"]["pinned_full_sha"]
    assert len(sha) == 40
    out["sha"] = sha
    out["hf_hub_version"] = huggingface_hub.__version__
    major = int(huggingface_hub.__version__.split(".")[0])
    if major < 2:
        _tag(q, "FAIL", f"huggingface_hub {huggingface_hub.__version__} < 2.0")
        ok = False
    _rec(q, f"huggingface_hub={huggingface_hub.__version__} (importable; >=2.0 {major >= 2})")

    from mlx_audio.utils import DEFAULT_ALLOW_PATTERNS

    out["mlx_default_allow_patterns_match"] = list(DEFAULT_ALLOW_PATTERNS) == ALLOW_PATTERNS
    _rec(
        q,
        f"ALLOW_PATTERNS mirrors mlx_audio DEFAULT_ALLOW_PATTERNS: {out['mlx_default_allow_patterns_match']}",
    )

    info = HfApi().model_info(MODEL_REPO, revision=sha, files_metadata=True, token=False)
    files = {s.rfilename: getattr(s, "size", None) for s in info.siblings}
    total = sum(v or 0 for v in files.values())
    out["files"] = files
    out["total_bytes_all"] = total
    import fnmatch

    matched = {n: s for n, s in files.items() if any(fnmatch.fnmatch(n, p) for p in ALLOW_PATTERNS)}
    out["matched_files"] = matched
    out["matched_bytes"] = sum(v or 0 for v in matched.values())
    _rec(q, f"files at {sha[:10]}: {len(files)} total {total / 1e6:.1f} MB")
    for n, s in sorted(files.items()):
        _rec(q, f"   {n}  {s}  {'(allowed)' if n in matched else '(NOT matched)'}")
    _rec(
        q,
        f"allow-patterns match {len(matched)}/{len(files)} files, {out['matched_bytes'] / 1e6:.1f} MB",
    )

    t0 = time.perf_counter()
    local = snapshot_download(MODEL_REPO, revision=sha, allow_patterns=ALLOW_PATTERNS)
    out["snapshot_path"] = local
    out["snapshot_secs"] = time.perf_counter() - t0
    on_disk = sorted(p.name for p in Path(local).iterdir())
    out["snapshot_listing"] = on_disk
    _rec(
        q,
        f"snapshot_download(revision=sha) -> {local} ({out['snapshot_secs']:.2f}s) files={on_disk}",
    )
    if sha not in Path(local).parts:
        _tag(q, "FAIL", "snapshot path does not contain the full sha dir")
        ok = False

    t0 = time.perf_counter()
    model = load(local, lazy=False, strict=True)
    out["load_secs"] = time.perf_counter() - t0
    out["sample_rate"] = int(model.sample_rate)
    out["model_class"] = type(model).__name__
    _rec(
        q,
        f"load(local, lazy=False, strict=True): {out['model_class']} rate={out['sample_rate']} ({out['load_secs']:.2f}s)",
    )
    if out["sample_rate"] != 24000:
        _tag(q, "FAIL", f"rate {out['sample_rate']} != 24000")
        ok = False

    bogus = "0" * 40
    try:
        snapshot_download(MODEL_REPO, revision=bogus, allow_patterns=ALLOW_PATTERNS)
        out["bogus_sha"] = "NO ERROR"
        _tag(q, "FAIL", "bogus sha did not raise")
        ok = False
    except Exception as exc:  # noqa: BLE001
        out["bogus_sha"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    _rec(q, f"bogus sha -> {out['bogus_sha']}")

    # Offline behaviour, complete vs partial snapshot (subprocess: HF_HUB_OFFLINE is read at import)
    out["offline_complete"] = _offline_snapshot(sha, "", ALLOW_PATTERNS)
    _rec(
        q,
        f"HF_HUB_OFFLINE=1 + complete cache -> ok={out['offline_complete']['ok']} {out['offline_complete'].get('err', '')}",
    )
    if not out["offline_complete"]["ok"]:
        _tag(q, "FAIL", "offline snapshot_download failed on a complete cache")
        ok = False
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(MODEL_REPO, revision=sha, allow_patterns=["config.json"], cache_dir=tmp)
        out["offline_partial"] = _offline_snapshot(sha, tmp, ALLOW_PATTERNS)
    _rec(
        q,
        "HF_HUB_OFFLINE=1 + PARTIAL cache (config.json only) -> "
        f"ok={out['offline_partial']['ok']} "
        f"{out['offline_partial'].get('err') or out['offline_partial'].get('files')}",
    )
    res[q] = out
    _tag(q, "PASS" if ok else "FAIL", f"pin mechanism (sha {sha})")
    return ("PASS" if ok else "FAIL"), model, local


# --------------------------------------------------------------------------- #
# Q3
# --------------------------------------------------------------------------- #
def q3_signature(res: dict, model) -> str:
    import inspect

    from tts_server.backends._introspect_util import verify_generate_signature

    q = "q3"
    out: dict = {}
    ok = True
    sig = inspect.signature(model.generate)
    params = {
        n: (None if p.default is inspect._empty else p.default, p.kind.name)
        for n, p in sig.parameters.items()
    }
    out["signature"] = str(sig)
    _rec(q, f"generate{sig}")
    for forbidden in ("ref_audio", "frames_after_eos"):
        present = forbidden in params
        _rec(q, f"{forbidden} in signature: {present} (never forwarded by the backend)")
        out[f"{forbidden}_present"] = present
        if not present:
            _tag(q, "RECORD", f"{forbidden} absent — negative guard is vacuous for it")
    out["has_var_keyword"] = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    explicit = [n for n in params if n not in ("self", "text", "kwargs")]
    out["explicit_params"] = explicit
    _rec(
        q,
        f"explicit params: {explicit}; **kwargs also reads lsd_decode_steps/noise_clamp/eos_threshold (source-verified)",
    )
    out["advertise_extras"] = ["temperature"] if "temperature" in params else []
    _rec(q, f"extras to advertise: {out['advertise_extras']}")
    if "temperature" not in params:
        ok = False

    # voice=None omit rule: omitting the kwarg must synthesize with the default voice
    r = _timed_generate(model, "Hello there.")  # no voice kwarg
    out["voice_omitted_ok"] = r["audio_s"] > 0
    _rec(q, f"voice kwarg omitted -> {r['audio_s']:.2f}s audio ({r['n_chunks']} chunks)")
    gen = model.generate(
        "Hello there.", voice="default", stream=True, streaming_interval=CANDIDATE_INTERVAL
    )
    n = sum(1 for _ in gen)
    out["voice_default_literal_ok"] = n > 0
    _rec(
        q,
        f"voice='default' literal also works ({n} chunks); backend still OMITS the kwarg for 'default'",
    )
    if not out["voice_omitted_ok"]:
        ok = False

    # verify_generate_signature drift warning (log-only)
    class _Drift:
        def generate(self, text, voice=None, **_kw):  # **_kw would satisfy; use no-varkw below
            return iter(())

    class _DriftStrict:
        def generate(self, text, voice=None):  # no temperature, no **kwargs
            return iter(())

    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, rec):  # noqa: D401
            records.append(rec)

    lg = logging.getLogger("phonon_gate_probe")
    lg.addHandler(_H())
    lg.setLevel(logging.WARNING)
    verify_generate_signature(_DriftStrict(), ["temperature"], "phonon", logger=lg)
    warned_strict = any("temperature" in r.getMessage() for r in records)
    records.clear()
    verify_generate_signature(_Drift(), ["temperature"], "phonon", logger=lg)
    warned_varkw = bool(records)
    records.clear()
    verify_generate_signature(model, ["temperature"], "phonon", logger=lg)
    warned_real = bool(records)
    out["verify_warns_on_drift"] = warned_strict
    out["verify_silent_with_varkw"] = not warned_varkw
    out["verify_silent_on_real_model"] = not warned_real
    _rec(
        q,
        f"verify_generate_signature: warns on drift={warned_strict}, silent with **kwargs="
        f"{not warned_varkw}, silent on real model={not warned_real} (log-only; merge_extras allowlist enforces)",
    )
    if not (warned_strict and not warned_real):
        ok = False
    res[q] = out
    _tag(q, "PASS" if ok else "FAIL", "signature / extras / omit rule / drift guard")
    return "PASS" if ok else "FAIL"


# --------------------------------------------------------------------------- #
# Q4
# --------------------------------------------------------------------------- #
def q4_voices(res: dict, model, out_dir: Path) -> str:
    import mlx.core as mx
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    q = "q4"
    out: dict = {}
    ok = True
    api = HfApi()

    vinfo = api.model_info(VOICES_REPO, token=False)
    vsha = vinfo.sha
    out["voices_sha"] = vsha
    out["voices_gated"] = vinfo.gated
    out["model_gated_unauth"] = res["q1"]["pinned_gated"]
    _rec(
        q,
        f"voices repo sha={vsha} gated={vinfo.gated} (model repo gated={res['q1']['pinned_gated']})",
    )
    if vinfo.gated is not False or res["q1"]["pinned_gated"] is not False:
        _tag(q, "FAIL", "a repo is gated (unauthenticated)")
        ok = False

    rows = _fetch_catalog(vsha)
    out["catalog_count"] = len(rows)
    out["catalog_keys"] = sorted({k for r in rows for k in r})
    _rec(q, f"voices.jsonl: {len(rows)} rows, keys={out['catalog_keys']}")
    langs: dict = {}
    for r in rows:
        langs[r.get("language")] = langs.get(r.get("language"), 0) + 1
    out["per_language"] = langs
    _rec(q, f"per-language counts: {langs}")
    name_key = "name"
    id_key = "voice_id"
    slugs = [_slug(r[name_key]) for r in rows]
    naive = [_slug(r[name_key], collapse=False) for r in rows]
    out["slug_unique"] = len(set(slugs)) == len(slugs)
    out["slug_naive_unique"] = len(set(naive)) == len(naive)
    out["slug_charset_ok"] = all(re.fullmatch(r"[a-z0-9_]+", s) for s in slugs)
    out["ids_unique"] = len({r[id_key] for r in rows}) == len(rows)
    out["slug_has_default"] = "default" in slugs
    dup = sorted({s for s in slugs if slugs.count(s) > 1})
    _rec(
        q,
        f"slug (NFKD, lower, runs of non-alnum -> '_', strip '_'): unique={out['slug_unique']} "
        f"charset={out['slug_charset_ok']} ids_unique={out['ids_unique']} 'default'-collision={out['slug_has_default']} "
        f"dups={dup}; naive per-char variant unique={out['slug_naive_unique']}",
    )
    for r, s in list(zip(rows, slugs))[:6]:
        _rec(
            q,
            f"   e.g. {r[name_key]!r} -> {s!r} ({r[id_key]}, {r.get('language')}/{r.get('locale')}, {r.get('gender')})",
        )
    if not (
        out["slug_unique"]
        and out["slug_charset_ok"]
        and out["ids_unique"]
        and not out["slug_has_default"]
    ):
        _tag(q, "FAIL", "slug rule not unique/valid")
        ok = False
    by_slug = {s: r for s, r in zip(slugs, rows)}
    out["has_alba"] = "alba" in by_slug
    _rec(q, f"catalog contains slug 'alba': {out['has_alba']}")

    # Download-cost measurements into isolated caches (cold)
    def _file(r):
        return f"voices/{r[id_key]}.safetensors"

    cost = {}
    for n in (1, 5, 37):
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.perf_counter()
            tot = 0
            for r in rows[:n]:
                p = hf_hub_download(
                    VOICES_REPO, _file(r), revision=vsha, token=False, cache_dir=tmp
                )
                tot += os.path.getsize(p)
            cost[n] = {"secs": time.perf_counter() - t0, "bytes": tot}
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        snapshot_download(
            VOICES_REPO,
            revision=vsha,
            token=False,
            cache_dir=tmp,
            allow_patterns=["voices/*.safetensors"],
        )
        tot = sum(
            os.path.getsize(p) for p in Path(tmp).rglob("*.safetensors")
        )  # follows blob symlinks
        cost["all"] = {"secs": time.perf_counter() - t0, "bytes": tot}
    out["download_cost"] = {str(k): v for k, v in cost.items()}
    for k, v in cost.items():
        _rec(
            q,
            f"per-file download N={k}: {v['secs']:.2f}s {v['bytes'] / 1e6:.2f} MB (cold, unauthenticated)",
        )

    # Warm local cache for the rest
    snap = snapshot_download(
        VOICES_REPO,
        revision=vsha,
        token=False,
        allow_patterns=["voices/*.safetensors", "voices.jsonl"],
    )
    paths = {s: str(Path(snap) / _file(r)) for s, r in by_slug.items()}
    # warm-cache hf_hub_download(local_files_only=True) call time
    t0 = time.perf_counter()
    p1 = hf_hub_download(VOICES_REPO, _file(rows[0]), revision=vsha, local_files_only=True)
    out["warm_local_files_only_secs"] = time.perf_counter() - t0
    _rec(
        q,
        f"warm hf_hub_download(local_files_only=True): {out['warm_local_files_only_secs'] * 1000:.1f} ms -> abs={os.path.isabs(p1)} isfile={os.path.isfile(p1)}",
    )
    try:
        hf_hub_download(
            VOICES_REPO, "voices/does-not-exist.safetensors", revision=vsha, local_files_only=True
        )
        out["local_files_only_miss"] = "NO ERROR"
        ok = False
    except Exception as exc:  # noqa: BLE001
        out["local_files_only_miss"] = type(exc).__name__
    _rec(q, f"local_files_only cache miss raises: {out['local_files_only_miss']}")

    # Load EVERY voice through the model's own voice path
    bad = []
    shapes = set()
    t0 = time.perf_counter()
    for s, pth in paths.items():
        try:
            w = mx.load(pth)
            key = "emb" if "emb" in w else "audio_prompt" if "audio_prompt" in w else list(w)[0]
            shapes.add((tuple(w[key].shape), key))
            prompt = model._resolve_audio_prompt(pth, None, False)
            st = model.get_state_for_audio_prompt(prompt)
            mx.eval(st if isinstance(st, dict) else [])
        except Exception as exc:  # noqa: BLE001
            bad.append((s, f"{type(exc).__name__}: {str(exc)[:100]}"))
    out["load_all"] = {
        "count": len(paths),
        "failed": bad,
        "shapes": [list(map(str, x)) for x in shapes],
        "secs": time.perf_counter() - t0,
    }
    _rec(
        q,
        f"load every voice via get_state_for_audio_prompt: {len(paths) - len(bad)}/{len(paths)} ok in {out['load_all']['secs']:.1f}s shapes={sorted(shapes)} failed={bad}",
    )
    if bad:
        _tag(q, "FAIL", "some catalog voices do not load at the pinned model revision")
        ok = False

    # absolute path accepted + bare-name / hf:// behaviour
    en = [s for s, r in by_slug.items() if r.get("language") == "en"]
    out["english_slugs"] = en
    r_def = _timed_generate(model, TEXT_SHORT)
    # pick: 2 female + 2 male english
    pick = []
    for g in ("female", "male"):
        pick += [s for s in en if str(by_slug[s].get("gender", "")).lower().startswith(g[0])][:2]
    pick = pick or en[:4]
    out["synth_picks"] = pick
    wavs = {}
    _write_wav(out_dir / "q4_default.wav", r_def["audio"], int(model.sample_rate))
    wavs["default"] = str(out_dir / "q4_default.wav")
    _audio_sanity("default", r_def["audio"], int(model.sample_rate))
    for s in pick:
        r = _timed_generate(model, TEXT_SHORT, voice=paths[s])
        f = out_dir / f"q4_{s}.wav"
        _write_wav(f, r["audio"], int(model.sample_rate))
        wavs[s] = str(f)
        _audio_sanity(s, r["audio"], int(model.sample_rate))
        _rec(
            q,
            f"named voice {s!r} ({by_slug[s].get('gender')}, {by_slug[s].get('locale')}): {r['audio_s']:.2f}s audio, TTFB {r['ttfb'] * 1000:.0f} ms, RTF {r['rtf']:.3f}",
        )
        if r["audio_s"] <= 0:
            ok = False
    out["wavs"] = wavs

    # one non-English voice (caveat): French voice, French text and English text
    fr = [s for s, r in by_slug.items() if r.get("language") == "fr"]
    if fr:
        s = fr[0]
        for label, text in (
            ("fr_text", "Bonjour, ceci est un court test de la voix. Il fait beau aujourd'hui."),
            ("en_text", TEXT_SHORT),
        ):
            r = _timed_generate(model, text, voice=paths[s])
            f = out_dir / f"q4_nonenglish_{s}_{label}.wav"
            _write_wav(f, r["audio"], int(model.sample_rate))
            wavs[f"{s}_{label}"] = str(f)
            _audio_sanity(f"{s}/{label}", r["audio"], int(model.sample_rate))
            _rec(
                q,
                f"non-English voice {s!r} with {label}: {r['audio_s']:.2f}s audio (LISTEN: model is -en)",
            )
    out["non_english_voice"] = fr[0] if fr else None

    # bare unknown name / bare catalog name / hf:// string
    def _try(v):
        try:
            pr = model._resolve_audio_prompt(v, None, False)
            return f"returned {pr!r}"
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {str(exc)[:90]}"

    probes = {
        "unknown bare 'zzz-not-a-voice'": _try("zzz-not-a-voice"),
        "bare catalog slug 'alba'": _try("alba"),
        f"bare catalog name {rows[0][name_key]!r}": _try(rows[0][name_key]),
        "hf:// string (unvalidated -> would be fetched)": _try(
            "hf://example/repo/voices/x.safetensors"
        ),
        "absolute local path": _try(paths[pick[0]]),
    }
    out["resolution_probes"] = probes
    for k, v in probes.items():
        _rec(q, f"_resolve_audio_prompt({k}) -> {v}")
    if not str(probes["unknown bare 'zzz-not-a-voice'"]).startswith("ValueError"):
        _tag(
            q,
            "RECORD",
            "unknown bare name did NOT raise ValueError — path-only resolution is the only guard",
        )
    if not str(probes["absolute local path"]).startswith("returned"):
        ok = False

    # warm named-voice TTFB (median of 5) vs default (median of 5)
    s0 = pick[0]
    d = [_timed_generate(model, TEXT_SHORT)["ttfb"] for _ in range(5)]
    nm = [_timed_generate(model, TEXT_SHORT, voice=paths[s0])["ttfb"] for _ in range(5)]
    out["ttfb_default_med_ms"] = _med(d) * 1000
    out["ttfb_named_warm_med_ms"] = _med(nm) * 1000
    _rec(
        q,
        f"warm-cache TTFB median of 5 @ {CANDIDATE_INTERVAL}s: default {out['ttfb_default_med_ms']:.0f} ms, named({s0}) {out['ttfb_named_warm_med_ms']:.0f} ms",
    )

    out["voices_catalog_rows"] = [
        {
            "slug": s,
            "id": r[id_key],
            "name": r[name_key],
            "language": r.get("language"),
            "locale": r.get("locale"),
            "gender": r.get("gender"),
        }
        for s, r in zip(slugs, rows)
    ]
    res[q] = out
    res["_paths"] = paths
    res["_by_slug"] = {s: r for s, r in by_slug.items()}
    _tag(q, "PASS" if ok else "FAIL", f"voices (voices sha {vsha})")
    return "PASS" if ok else "FAIL"


# --------------------------------------------------------------------------- #
# Q5
# --------------------------------------------------------------------------- #
def q5_interval(res: dict, model) -> str:
    q = "q5"
    out: dict = {}
    ok = True
    pick = res["q4"]["synth_picks"][0]
    named = res["_paths"][pick]
    _timed_generate(model, TEXT_SHORT)  # warm
    table = {}
    for iv in (0.3, 0.5, 1.0, 2.0):
        for label, voice in (("default", None), ("named", named)):
            runs = [_timed_generate(model, TEXT_SHORT, voice=voice, interval=iv) for _ in range(3)]
            row = {
                "ttfb_ms": _med([r["ttfb"] for r in runs]) * 1000,
                "rtf": _med([r["rtf"] for r in runs]),
                "n_chunks": runs[0]["n_chunks"],
                "max_gap_ms": max(r["max_gap"] for r in runs) * 1000,
                "gap_violations": sum(r["gap_violations"] for r in runs),
                "first_chunk_s": runs[0]["chunk_durs"][0] if runs[0]["chunk_durs"] else None,
            }
            table[f"{iv}/{label}"] = row
            _rec(
                q,
                f"interval={iv:<3} {label:<7} TTFB {row['ttfb_ms']:6.0f} ms  RTF {row['rtf']:.3f}  "
                f"chunks {row['n_chunks']:2d}  max_gap {row['max_gap_ms']:6.0f} ms  gap>chunk_dur: {row['gap_violations']}",
            )
            if row["gap_violations"]:
                ok = False
    out["table"] = table
    # lock rule: smallest interval with zero continuity violations (both voices)
    lock = next(
        (
            iv
            for iv in (0.3, 0.5, 1.0, 2.0)
            if not table[f"{iv}/default"]["gap_violations"]
            and not table[f"{iv}/named"]["gap_violations"]
        ),
        None,
    )
    out["recommended_streaming_interval"] = lock
    _rec(
        q,
        f"recommended _STREAMING_INTERVAL = {lock} (smallest interval with zero continuity violations; also lowest TTFB if RTF<<1)",
    )
    res[q] = out
    _tag(q, "PASS" if ok and lock else "FAIL", "continuity: max inter-chunk gap <= chunk duration")
    return "PASS" if ok and lock else "FAIL"


# --------------------------------------------------------------------------- #
# Q6
# --------------------------------------------------------------------------- #
def q6_licence(res: dict) -> str:
    from huggingface_hub import HfApi, hf_hub_download

    q = "q6"
    out: dict = {}
    api = HfApi()
    for repo, rev in (
        (MODEL_REPO, res["q1"]["pinned_full_sha"]),
        (MODEL_REPO, "main"),
        (VOICES_REPO, res["q4"]["voices_sha"]),
    ):
        info = api.model_info(repo, revision=rev, token=False)
        card = getattr(info, "card_data", None)
        lic = getattr(card, "license", None) if card else None
        lic_name = getattr(card, "license_name", None) if card else None
        lic_link = getattr(card, "license_link", None) if card else None
        key = f"{repo}@{rev[:10]}"
        out[key] = {
            "license": lic,
            "license_name": lic_name,
            "license_link": lic_link,
            "gated": info.gated,
            "tags": [t for t in info.tags if "licen" in t],
        }
        _rec(
            q,
            f"{key}: license={lic!r} name={lic_name!r} link={lic_link!r} gated={info.gated} tags={out[key]['tags']}",
        )
        try:
            readme = Path(hf_hub_download(repo, "README.md", revision=rev, token=False)).read_text()
            lines = [
                ln.strip()
                for ln in readme.splitlines()
                if re.search(r"licen[sc]e", ln, re.IGNORECASE)
            ]
            out[key]["readme_licence_lines"] = lines[:12]
            for ln in lines[:12]:
                _rec(q, f"   README: {ln[:200]}")
        except Exception as exc:  # noqa: BLE001
            _rec(q, f"   README fetch failed: {type(exc).__name__}")
        files = [
            s.rfilename
            for s in info.siblings
            if "licen" in s.rfilename.lower() or s.rfilename.upper().startswith("NOTICE")
        ]
        out[key]["licence_files"] = files
        _rec(q, f"   licence/notice files in repo: {files or 'none'}")
        for f in files[:2]:
            txt = (
                Path(hf_hub_download(repo, f, revision=rev, token=False))
                .read_text()[:400]
                .replace("\n", " ")
            )
            _rec(q, f"   {f}: {txt}")

    # Sibling repos (40M etc.) — recorded only because checked here
    try:
        repos = [m.id for m in api.list_models(author="gradium", limit=50, token=False)]
        out["gradium_repos"] = repos
        _rec(q, f"gradium repos visible unauthenticated: {repos}")
        for rid in repos:
            if rid in (MODEL_REPO, VOICES_REPO) or "phonon" not in rid.lower():
                continue
            i = api.model_info(rid, token=False)
            c = getattr(i, "card_data", None)
            out[f"sibling:{rid}"] = {
                "license": getattr(c, "license", None) if c else None,
                "gated": i.gated,
            }
            _rec(q, f"sibling {rid}: license={out[f'sibling:{rid}']['license']!r} gated={i.gated}")
    except Exception as exc:  # noqa: BLE001
        _rec(q, f"sibling listing failed: {type(exc).__name__}: {str(exc)[:120]}")
    res[q] = out
    _tag(q, "RECORD", "docs may quote only the verified text above")
    return "RECORD"


# --------------------------------------------------------------------------- #
# Q7
# --------------------------------------------------------------------------- #
def q7_cancel(res: dict, model) -> str:
    q = "q7"
    out: dict = {}
    ok = True
    named = res["_paths"][res["q4"]["synth_picks"][0]]
    long_text = _story_prefix(600)  # ~100 words -> many chunks at 0.3 s
    for label, voice in (("default", None), ("named", named)):
        base = _timed_generate(model, TEXT_MULTI, voice=voice)
        for close_after in (1, 2, 5):
            part = _timed_generate(model, long_text, voice=voice, close_after=close_after)
            follow = _timed_generate(model, TEXT_MULTI, voice=voice)
            sane = bool(
                follow["audio_s"] > 0
                and not _has_nan(follow["audio"])
                and abs(follow["audio_s"] - base["audio_s"]) / max(base["audio_s"], 1e-6) < 0.35
            )
            if close_after > part["n_chunks"] and part["n_chunks"] < close_after:
                note = f"(only {part['n_chunks']} chunks available)"
            else:
                note = ""
            out[f"{label}/close_after_{close_after}"] = {
                "cancelled_chunks": part["n_chunks"],
                "follow_up_s": follow["audio_s"],
                "baseline_s": base["audio_s"],
                "follow_up_ttfb_ms": follow["ttfb"] * 1000,
                "sane": sane,
            }
            _rec(
                q,
                f"{label:<7} close after {close_after} chunk(s) {note}: follow-up {follow['audio_s']:.2f}s "
                f"(baseline {base['audio_s']:.2f}s) TTFB {follow['ttfb'] * 1000:.0f} ms sane={sane}",
            )
            ok = ok and sane
    # cross-voice reuse after a cancel
    _timed_generate(model, long_text, voice=named, close_after=2)
    cross = _timed_generate(model, TEXT_MULTI)
    out["cross_voice_follow_up_ok"] = cross["audio_s"] > 0 and not _has_nan(cross["audio"])
    _rec(q, f"named-cancel then default synth ok={out['cross_voice_follow_up_ok']}")
    ok = ok and out["cross_voice_follow_up_ok"]
    res[q] = out
    _tag(q, "PASS" if ok else "FAIL", "generator close at chunk boundary leaves the model healthy")
    return "PASS" if ok else "FAIL"


def _has_nan(a) -> bool:
    import numpy as np

    return bool(np.isnan(a).any())


# --------------------------------------------------------------------------- #
# Q8
# --------------------------------------------------------------------------- #
def q8_long_text(res: dict, model, out_dir: Path) -> str:
    q = "q8"
    out: dict = {}
    ok = True
    rows = {}
    n_sent = len(_STORY_SENTENCES)
    n_unique = len(set(_STORY_SENTENCES))
    full = _story_prefix(10**9)
    out["story"] = {
        "sentences": n_sent,
        "unique": n_unique,
        "chars": len(full),
        "words": len(full.split()),
    }
    _rec(
        q,
        f"story: {n_sent} sentences ({n_unique} unique), {len(full)} chars, {len(full.split())} words",
    )
    if n_unique != n_sent:
        _tag(q, "FAIL", "story contains repeated sentences (looped text)")
        ok = False
    ref = _timed_generate(model, _story_prefix(300))
    ref_words = len(_story_prefix(300).split())
    ref_spw = ref["audio_s"] / ref_words
    out["reference"] = {
        "chars": len(_story_prefix(300)),
        "words": ref_words,
        "audio_s": ref["audio_s"],
        "sec_per_word": ref_spw,
    }
    _rec(
        q,
        f"reference (~300 chars): {ref_words} words -> {ref['audio_s']:.1f}s audio, {ref_spw:.3f} s/word",
    )
    for n in (500, 1000, 2000, 3000, 4000, 10**9):  # 10**9 = the full story
        text = _story_prefix(n)
        words = len(text.split())
        r = _timed_generate(model, text)
        spw = r["audio_s"] / words
        ratio = spw / ref_spw
        rows[n] = {
            "chars": len(text),
            "words": words,
            "audio_s": r["audio_s"],
            "sec_per_word": spw,
            "ratio_vs_ref": ratio,
            "rtf": r["rtf"],
            "ttfb_ms": r["ttfb"] * 1000,
            "chunks": r["n_chunks"],
        }
        _rec(
            q,
            f"{len(text):>5} chars {words:>4} words -> {r['audio_s']:6.1f}s audio, {spw:.3f} s/word "
            f"(ratio {ratio:.2f}) RTF {r['rtf']:.3f} TTFB {r['ttfb'] * 1000:.0f} ms",
        )
        if n in (2000, 4000):
            _write_wav(out_dir / f"q8_story_{n}.wav", r["audio"], int(model.sample_rate))
            _audio_sanity(f"story{n}", r["audio"], int(model.sample_rate))
        if n <= 2000 and not (0.7 <= ratio <= 1.3):
            ok = False
    out["rows"] = {str(k): v for k, v in rows.items()}
    # marginal s/word of each increment (cumulative ratios can hide a missing tail)
    prev = None
    for n, v in rows.items():
        if prev is not None and v["words"] > prev["words"]:
            marg = (v["audio_s"] - prev["audio_s"]) / (v["words"] - prev["words"])
            v["marginal_sec_per_word"] = marg
            _rec(
                q,
                f"   increment to {v['chars']} chars: {marg:.3f} s/word over +{v['words'] - prev['words']} words",
            )
        prev = v
    first_bad = next((n for n, v in rows.items() if not (0.7 <= v["ratio_vs_ref"] <= 1.3)), None)
    out["first_truncation_chars"] = first_bad
    _rec(q, f"first size with s/word outside [0.7,1.3]x reference: {first_bad}")
    # ideal_words support: TTFB vs chunk size on the default voice
    ttfb = {}
    for w in (10, 40, 100):
        words = " ".join(_story_prefix(2000).split()[:w])
        ttfb[w] = _med([_timed_generate(model, words)["ttfb"] for _ in range(3)]) * 1000
    out["ttfb_by_words_ms"] = ttfb
    _rec(q, f"TTFB by commit size (default voice, @{CANDIDATE_INTERVAL}s): {ttfb}")
    res[q] = out
    _tag(
        q,
        "PASS" if ok else "FAIL",
        "duration scales with word count (unique story) up to 2000 chars"
        + ("" if first_bad is None else f"; outlier at {first_bad}"),
    )
    return "PASS" if ok else "FAIL"


# --------------------------------------------------------------------------- #
_QUESTIONS = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default=None, help="WAV/results.json dir (default: fresh mkdtemp)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="phonon-gate-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out dir: {out_dir}")

    try:
        import mlx.core as mx
        import mlx_audio  # noqa: F401
    except ImportError as exc:
        print(
            f"FATAL: mlx_audio not importable ({exc}). Run via: uv run --extra pocket_tts python {sys.argv[0]}"
        )
        return 2
    mx.random.seed(args.seed)

    res: dict = {}
    verdicts: dict = {}

    def _run(name, fn, *a):
        try:
            return fn(*a)
        except Exception as exc:  # noqa: BLE001 - gate records, never crashes the sequence
            import traceback

            traceback.print_exc()
            _tag(name, "FAIL", f"exception {type(exc).__name__}: {str(exc)[:200]}")
            return "FAIL"

    verdicts["q1"] = _run("q1", q1_durability, res)
    print()
    r2 = _run("q2", q2_pin, res)
    model = None
    if isinstance(r2, tuple):
        verdicts["q2"], model, _local = r2
    else:
        verdicts["q2"] = r2
    print()
    if model is None:
        print("FATAL: pinned model did not load; remaining questions skipped")
        for qn in _QUESTIONS[2:]:
            verdicts[qn] = "SKIP"
    else:
        verdicts["q3"] = _run("q3", q3_signature, res, model)
        print()
        verdicts["q4"] = _run("q4", q4_voices, res, model, out_dir)
        print()
        have_voices = "_paths" in res
        verdicts["q5"] = _run("q5", q5_interval, res, model) if have_voices else "SKIP"
        print()
        verdicts["q6"] = _run("q6", q6_licence, res) if have_voices else "SKIP"
        print()
        verdicts["q7"] = _run("q7", q7_cancel, res, model) if have_voices else "SKIP"
        print()
        verdicts["q8"] = _run("q8", q8_long_text, res, model, out_dir)
        print()

    res.pop("_paths", None)
    res.pop("_by_slug", None)
    for k, v in res.items():
        if isinstance(v, dict):
            for kk in list(v):
                if kk == "audio":
                    v.pop(kk)
    (out_dir / "results.json").write_text(
        json.dumps({"verdicts": verdicts, "results": res}, indent=2, default=str)
    )
    print(f"results.json: {out_dir / 'results.json'}")
    summary = " ".join(f"{q.upper()}={verdicts.get(q, 'SKIP')}" for q in _QUESTIONS)
    print(f"GATE-SUMMARY {summary}")
    failed = [q for q, v in verdicts.items() if v == "FAIL"]
    if failed:
        print(f"phonon phase0 gate: FAILED ({', '.join(q.upper() for q in failed)}) — re-plan")
        return 1
    print("phonon phase0 gate: PASS (RECORD items go into the plan's ## Findings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
