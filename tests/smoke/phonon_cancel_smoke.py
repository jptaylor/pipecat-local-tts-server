#!/usr/bin/env python3
"""Phonon server-level smoke: named voice, voice validation, barge-in, disconnect.

Starts its OWN server (mktemp socket, never the operator's canonical one) with
``PIPECAT_TTS_PHONON_VOICES=<english slug>,<typo'd slug>`` and asserts, through the
real wire path:

  1. ``server.status`` voices == ``["default", <slug>]`` (typo skipped, server up);
  2. audio for ``voice=<slug>``;
  3. ``invalid_config`` for ``voice="hf://x"`` and for the typo'd slug;
  4. barge-in: ``response.cancel`` mid-stream -> ``response.cancelled``, no further
     deltas for that response, and a follow-up commit returns full audio (default
     voice and the named voice);
  5. abrupt disconnect mid-generate (TCP abort, no close frame), then a fresh
     connection still synthesizes full audio (shared model not poisoned).

Usage (needs the phonon extra and a warm HF cache or network):
  uv run --extra phonon python tests/smoke/phonon_cancel_smoke.py [--voice zoey]
Exit code 0 only if every assertion holds.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tts_server.client import TTSClient

_LONG = (
    "This is a deliberately long passage so that synthesis is still running when the "
    "client interrupts it. It goes on for several sentences. The quick brown fox jumps "
    "over the lazy dog, and then it keeps on running across the field for quite a while, "
    "past the barn, the river and the old stone bridge."
)
_SHORT = "The quick brown fox jumps over the lazy dog."
_TYPO = "no_such_voice_zzz"
_MIN_SAMPLES = 12_000  # 0.5 s at 24 kHz: a "full" short utterance is ~2-3 s


async def _drain(client: TTSClient, timeout: float) -> dict:
    """Read one response to done; return samples + the first error/failure if any."""
    samples = 0
    err = None

    async def _go() -> None:
        nonlocal samples, err
        async for ev in client.events():
            t = ev.get("type")
            if t == "response.audio.delta":
                samples += len(base64.b64decode(ev["audio"])) // 2
            elif t == "response.audio.done":
                return
            elif t in ("error", "response.failed"):
                err = ev
                return

    await asyncio.wait_for(_go(), timeout)
    return {"samples": samples, "error": err}


async def _synth(client: TTSClient, text: str, voice: str | None, timeout: float) -> dict:
    await client.append(text)
    await client.commit(voice=voice)
    return await _drain(client, timeout)


def _code(ev: dict | None) -> str | None:
    if not ev:
        return None
    e = ev.get("error")
    return e.get("code") if isinstance(e, dict) else None


async def _status(client: TTSClient, timeout: float) -> dict:
    await client.status()

    async def _go() -> dict:
        async for ev in client.events():
            if ev.get("type") == "server.status":
                return ev
        raise RuntimeError("connection closed before server.status")

    return await asyncio.wait_for(_go(), timeout)


async def _run(sock: str, voice: str, timeout: float) -> int:
    fails: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"   {'PASS' if ok else 'FAIL'}: {label}{' — ' + detail if detail else ''}")
        if not ok:
            fails.append(label)

    client = TTSClient(socket_path=sock)
    hello = await client.connect()
    rate = hello["audio"]["rate"]
    print(f"connected: backend={hello.get('backend')} rate={rate}")

    print("== status / voices ==")
    st = await _status(client, timeout)
    check(st.get("voices") == ["default", voice], "status voices", repr(st.get("voices")))

    print("== named voice audio ==")
    res = await _synth(client, _SHORT, voice, timeout)
    check(
        res["error"] is None and res["samples"] >= _MIN_SAMPLES,
        f"audio for voice={voice}",
        f"{res['samples']} samples",
    )

    print("== voice validation ==")
    for bad in ("hf://x", _TYPO):
        res = await _synth(client, _SHORT, bad, timeout)
        check(
            _code(res["error"]) == "invalid_config",
            f"invalid_config for voice={bad!r}",
            repr(_code(res["error"])),
        )
        await client.clear()

    print("== barge-in ==")
    for v in (None, voice):
        label = v or "default"
        await client.append(_LONG)
        await client.commit(voice=v)
        deltas = 0
        cancelled = False
        late = 0
        t_cancel = 0.0
        t_ack = None

        async def _go() -> None:
            nonlocal deltas, cancelled, late, t_cancel, t_ack
            async for ev in client.events():
                t = ev.get("type")
                if t == "response.audio.delta":
                    if cancelled:
                        late += 1
                        continue
                    deltas += 1
                    if deltas == 2:
                        t_cancel = time.monotonic()
                        await client.cancel()
                        cancelled = True
                elif t == "response.cancelled":
                    t_ack = time.monotonic()
                    return
                elif t == "response.audio.done":
                    return
                elif t in ("error", "response.failed"):
                    raise RuntimeError(f"unexpected {t}: {ev}")

        await asyncio.wait_for(_go(), timeout)
        check(
            cancelled and t_ack is not None,
            f"[{label}] response.cancelled received",
            f"after {deltas} deltas",
        )
        if t_ack is not None:
            print(
                f"      cancel->ack {(t_ack - t_cancel) * 1000:.1f} ms, deltas already in flight before ack={late}"
            )
        await asyncio.sleep(0.3)  # let any in-flight delta land
        res = await _synth(client, _SHORT, v, timeout)
        check(
            res["error"] is None and res["samples"] >= _MIN_SAMPLES,
            f"[{label}] follow-up after cancel",
            f"{res['samples']} samples",
        )
    await client.close()

    print("== abrupt disconnect then reuse ==")
    victim = TTSClient(socket_path=sock)
    await victim.connect()
    await victim.append(_LONG)
    await victim.commit(voice=voice)
    await asyncio.sleep(0.3)
    transport = getattr(victim._ws, "transport", None)
    if transport is None:
        check(False, "could not reach client transport to abort")
    else:
        transport.abort()
    await asyncio.sleep(1.0)
    fresh = TTSClient(socket_path=sock)
    await fresh.connect()
    try:
        for v in (None, voice):
            res = await _synth(fresh, _SHORT, v, timeout)
            check(
                res["error"] is None and res["samples"] >= _MIN_SAMPLES,
                f"fresh commit after abort [{v or 'default'}]",
                f"{res['samples']} samples",
            )
    except TimeoutError:
        check(False, "fresh commit after abort timed out (model poisoned/stalled)")
    finally:
        await fresh.close()

    print(f"== summary: {'PASS' if not fails else 'FAIL'} (failures={len(fails)}) ==")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--voice", default="zoey", help="English catalog slug to enable (default zoey)")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--startup-timeout", type=float, default=300.0)
    args = ap.parse_args()

    run_dir = Path(tempfile.mkdtemp(prefix="phonon-cancel."))
    sock = str(run_dir / "tts.sock")
    log_path = run_dir / "server.log"
    env = dict(os.environ, PIPECAT_TTS_PHONON_VOICES=f"{args.voice},{_TYPO}")
    log = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        [sys.executable, "-m", "tts_server", "serve", "--backend", "phonon", "--socket-path", sock],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    try:
        deadline = time.monotonic() + args.startup_timeout
        while not os.path.exists(sock):
            if proc.poll() is not None:
                print("server exited before listening; log tail:", file=sys.stderr)
                print("\n".join(log_path.read_text().splitlines()[-20:]), file=sys.stderr)
                return 1
            if time.monotonic() > deadline:
                print("socket never appeared", file=sys.stderr)
                return 1
            time.sleep(0.5)
        return asyncio.run(_run(sock, args.voice, args.timeout))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
