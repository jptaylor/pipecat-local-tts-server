"""Phonon weight-variant probe: safetensors vs q8 GGUF vs q4k GGUF.

Phase 0 follow-up (not a verdict gate). Answers: does mlx-audio's pocket_tts
loader load each variant from the pinned snapshot, and what do load time, peak
memory, TTFB and RTF look like for the default voice and a named voice (Zoey)?

Each variant runs in its own subprocess so load time and peak memory are clean.
GGUF variants are loaded from a scratch dir holding symlinks to the snapshot
files MINUS ``model.safetensors`` (the loader prefers it over a lone GGUF).

    uv run --extra pocket_tts python tests/smoke/phonon_quant_probe.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _gate_helpers import _write_wav  # noqa: E402
from phonon_phase0_gate import (  # noqa: E402
    ALLOW_PATTERNS,
    MODEL_REPO,
    PINNED_SHORT,
    VOICES_REPO,
    _story_prefix,
    _timed_generate,
)

# kind, arg: "st" = pinned model.safetensors; "mlxq" = quantize it locally with
# mlx-audio's pocket_tts convert() (arg = bits); "gguf" = the repo's prebuilt file.
VARIANTS = {
    "safetensors": ("st", None),
    "mlx_q8": ("mlxq", 8),
    "mlx_q4": ("mlxq", 4),
    "gguf_q8": ("gguf", "model.q8.gguf"),
    "gguf_q4k": ("gguf", "model.q4k.gguf"),
}
Q_GROUP_SIZE = 32
ZOEY_FILE = "voices/NbpkqMVS3CJeq2j8.safetensors"  # voices.jsonl row "Zoey"
SHORT_TEXT = "The train leaves at nine, so we should go."
REPS = 5
WARMUPS = 2


def _snapshot(variant: str) -> Path:
    from huggingface_hub import snapshot_download

    patterns = list(ALLOW_PATTERNS)
    kind, arg = VARIANTS[variant]
    if kind == "gguf":
        patterns.append(arg)
    return Path(snapshot_download(MODEL_REPO, revision=PINNED_SHORT, allow_patterns=patterns))


def _model_dir(variant: str, scratch: Path) -> Path:
    snap = _snapshot(variant)
    kind, _ = VARIANTS[variant]
    if kind == "st":
        return snap
    d = scratch / f"model_{variant}"
    if kind == "mlxq":
        from mlx_audio.tts.models.pocket_tts.convert import convert

        if not d.exists():
            convert(
                str(snap),
                str(d),
                quantize=True,
                q_bits=VARIANTS[variant][1],
                q_group_size=Q_GROUP_SIZE,
            )
        return d
    d.mkdir(parents=True, exist_ok=True)
    for f in snap.iterdir():
        if f.name == "model.safetensors":
            continue
        link = d / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    return d


def _run_variant(variant: str, out_dir: Path) -> dict:
    import mlx.core as mx
    from huggingface_hub import hf_hub_download
    from mlx_audio.tts.utils import load

    res: dict = {"variant": variant}
    tc = time.perf_counter()
    try:
        mdir = _model_dir(variant, out_dir / "scratch")
    except Exception as exc:  # noqa: BLE001
        res["prepare_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return res
    res["prepare_s"] = time.perf_counter() - tc
    res["disk_mb"] = sum(f.stat().st_size for f in mdir.rglob("*") if f.is_file()) / 1e6
    zoey = hf_hub_download(VOICES_REPO, ZOEY_FILE, token=False)
    medium = _story_prefix(260)

    t0 = time.perf_counter()
    try:
        model = load(str(mdir), lazy=False, strict=True)
    except Exception as exc:  # noqa: BLE001 - the probe's point is to record failures
        res["load_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return res
    res["load_s"] = time.perf_counter() - t0
    res["rate"] = int(model.sample_rate)
    res["mem_after_load_mb"] = mx.get_active_memory() / 1e6

    for vname, voice in (("default", None), ("zoey", zoey)):
        for tname, text in (("short", SHORT_TEXT), ("medium", medium)):
            runs = []
            try:
                for i in range(WARMUPS + REPS):
                    mx.random.seed(i)
                    r = _timed_generate(model, text, voice=voice)
                    if i >= WARMUPS:
                        runs.append(r)
            except Exception as exc:  # noqa: BLE001
                res[f"{vname}_{tname}"] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            ttfb = [r["ttfb"] for r in runs]
            rtf = [r["rtf"] for r in runs]
            aud = runs[-1]["audio"]
            import numpy as np

            res[f"{vname}_{tname}"] = {
                "ttfb_med_ms": statistics.median(ttfb) * 1000,
                "ttfb_min_ms": min(ttfb) * 1000,
                "ttfb_max_ms": max(ttfb) * 1000,
                "rtf_med": statistics.median(rtf),
                "audio_s": runs[-1]["audio_s"],
                "gap_violations": sum(r["gap_violations"] for r in runs),
                "nan": int(np.isnan(aud).sum()),
                "max_abs": float(np.max(np.abs(aud))) if aud.size else 0.0,
            }
            if tname == "medium":
                _write_wav(out_dir / f"quant_{variant}_{vname}.wav", aud, res["rate"])
    res["peak_mem_mb"] = mx.get_peak_memory() / 1e6
    return res


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", choices=list(VARIANTS))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.variant:  # child: one variant, JSON on the last stdout line
        print("RESULT " + json.dumps(_run_variant(args.variant, out_dir)))
        return 0

    results = []
    for v in VARIANTS:
        print(f"== {v} ==", flush=True)
        p = subprocess.run(
            [sys.executable, __file__, "--out-dir", str(out_dir), "--variant", v],
            capture_output=True,
            text=True,
            env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
        )
        line = next((s for s in reversed(p.stdout.splitlines()) if s.startswith("RESULT ")), None)
        if line is None:
            results.append({"variant": v, "crash": p.stderr[-800:]})
        else:
            results.append(json.loads(line[len("RESULT ") :]))
        print(json.dumps(results[-1], indent=1), flush=True)
    (out_dir / "quant_probe.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
