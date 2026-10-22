#!/usr/bin/env python3
"""End-to-end check of the reference Pipecat adapter against a real phonon server.

Starts its OWN phonon server on a mktemp socket (never the operator's canonical
one), runs ``examples/pipecat_tts_service.py::LocalTTSService`` inside a real
Pipecat pipeline, speaks two utterances with the DEFAULT voice and asserts on the
SECOND (warm) one: 24 kHz, >= 1 audio frame, first-audio-frame TTFB < 0.1 s. The
first utterance only has to succeed. Named voices are not asserted against 0.1 s.

Usage (needs both extras):
  uv run --extra examples --extra phonon python tests/smoke/pipecat_adapter_e2e.py
Exit code 0 only if every assertion holds.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.pipecat_tts_service import LocalTTSService

TEXT = "The quick brown fox jumps over the lazy dog."
EXPECTED_RATE = 24000
TTFB_BOUND = 0.1


class _Sink(FrameProcessor):
    """Records per-utterance timing: queue time -> first audio frame -> stopped."""

    def __init__(self) -> None:
        super().__init__()
        self.utterances: list[dict] = []
        self.errors: list[str] = []
        self.on_stopped: asyncio.Queue[None] = asyncio.Queue()
        self.t_sent = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        now = time.monotonic()
        if isinstance(frame, TTSStartedFrame):
            self.utterances.append({"first": None, "frames": 0, "rate": None, "bytes": 0})
        elif isinstance(frame, TTSAudioRawFrame) and self.utterances:
            u = self.utterances[-1]
            if u["first"] is None:
                u["first"] = now - self.t_sent
                u["rate"] = frame.sample_rate
            u["frames"] += 1
            u["bytes"] += len(frame.audio)
        elif isinstance(frame, ErrorFrame):
            self.errors.append(str(frame.error))
        elif isinstance(frame, TTSStoppedFrame):
            self.on_stopped.put_nowait(None)
        await self.push_frame(frame, direction)


async def _run(sock: str, timeout: float) -> int:
    sink = _Sink()
    tts = LocalTTSService(socket_path=sock)
    task = PipelineTask(Pipeline([tts, sink]))
    runner = PipelineRunner(handle_sigint=False)
    runner_task = asyncio.create_task(runner.run(task))
    try:
        for i in range(2):
            sink.t_sent = time.monotonic()
            await task.queue_frame(TTSSpeakFrame(TEXT))
            await asyncio.wait_for(sink.on_stopped.get(), timeout)
            u = sink.utterances[-1]
            first = u["first"]
            first_s = "n/a" if first is None else f"{first:.3f}s"
            print(f"utterance {i + 1}: frames={u['frames']} rate={u['rate']} ttfb={first_s}")
        await task.queue_frame(EndFrame())
        await asyncio.wait_for(runner_task, timeout)
    finally:
        if not runner_task.done():
            await task.cancel()
            await asyncio.gather(runner_task, return_exceptions=True)

    fails: list[str] = []
    if sink.errors:
        fails.append(f"error frames: {sink.errors}")
    if len(sink.utterances) < 2:
        fails.append(f"expected 2 utterances, got {len(sink.utterances)}")
    else:
        warm = sink.utterances[1]
        if warm["frames"] < 1:
            fails.append("warm utterance produced no audio frames")
        if warm["rate"] != EXPECTED_RATE:
            fails.append(f"warm utterance rate {warm['rate']} != {EXPECTED_RATE}")
        if warm["first"] is None or warm["first"] >= TTFB_BOUND:
            fails.append(f"warm first-frame TTFB {warm['first']} not < {TTFB_BOUND}s")
        if sink.utterances[0]["frames"] < 1:
            fails.append("first utterance produced no audio frames")
    for f in fails:
        print(f"FAIL: {f}")
    print(f"== summary: {'PASS' if not fails else 'FAIL'} ==")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--startup-timeout", type=float, default=300.0)
    args = ap.parse_args()

    run_dir = Path(tempfile.mkdtemp(prefix="phonon-adapter-e2e."))
    sock = str(run_dir / "tts.sock")
    log_path = run_dir / "server.log"
    log = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        [sys.executable, "-m", "tts_server", "serve", "--backend", "phonon", "--socket-path", sock],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(os.environ),
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
        return asyncio.run(_run(sock, args.timeout))
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
