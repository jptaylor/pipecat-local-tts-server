"""Phonon backend tests (MLX-GATED — Apple Silicon required).

NOT on the lean allow-list. Skipped only when ``mlx_audio`` is unavailable; on
an Apple Silicon host with a warm HF cache NOTHING here skips. Loads the pinned
revision of ``gradium/phonon-31m-en`` (via a local snapshot) once per module and
enables two catalog voices so the named-voice path is exercised for real.

    uv run --extra phonon pytest tests/test_phonon_backend.py -v
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx_audio")

from tts_server.backends import phonon as PH

from ._helpers import connected_client, running_server, synthesize_once

pytestmark = pytest.mark.asyncio

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STORY = _REPO_ROOT / "tests" / "smoke" / "fixtures" / "phonon" / "story_long.txt"
_SENTENCE = "The quick brown fox jumps over the lazy dog and then keeps on running for a while."
_VOICES_ENV = "PIPECAT_TTS_PHONON_VOICES"
_NAMED = "alba"  # a real catalog slug (not the pocket voice)


@pytest.fixture(scope="module")
async def started_backend():
    mp = pytest.MonkeyPatch()
    mp.setenv(_VOICES_ENV, _NAMED)
    backend = PH.PhononBackend()
    try:
        await backend.start()
        yield backend
    finally:
        await backend.close()
        mp.undo()


async def _drain(stream) -> tuple[int, int]:
    """Return (native delta chunks, total pcm bytes)."""
    n = nbytes = 0
    async for ev in stream.events():
        if ev.kind == "delta" and ev.pcm:
            n += 1
            nbytes += len(ev.pcm)
    return n, nbytes


async def test_real_load_rate_and_capabilities(started_backend):
    assert started_backend.sample_rate == 24000
    assert int(started_backend._loaded_model.sample_rate) == 24000
    caps = started_backend.capabilities()
    assert caps["streaming"] is True
    assert caps["extras"] == ["temperature"]
    assert caps["languages"] == ["en"]
    assert started_backend.voices()[0] == "default"
    assert _NAMED in started_backend.voices()
    assert caps["voice_count"] == len(started_backend.voices())


async def test_hello_advertises_rate_and_voices():
    mp = pytest.MonkeyPatch()
    mp.setenv(_VOICES_ENV, _NAMED)
    try:
        backend = PH.PhononBackend()
        async with running_server(backend) as srv, connected_client(srv) as (_c, hello):
            assert hello["audio"]["rate"] == 24000
            assert hello["capabilities"]["streaming"] is True
    finally:
        mp.undo()


async def test_short_synthesis_default_voice(started_backend):
    stream = await started_backend.open_stream(voice=None, language=None, extras=None)
    await stream.feed(_SENTENCE)
    await stream.end()
    chunks, nbytes = await _drain(stream)
    assert chunks >= 2, f"expected sub-segment streaming, got {chunks} chunks"
    seconds = nbytes / 2 / 24000
    assert 2.0 < seconds < 15.0


async def test_named_voice_synthesis(started_backend):
    stream = await started_backend.open_stream(voice=_NAMED, language=None, extras=None)
    await stream.feed(_SENTENCE)
    await stream.end()
    chunks, nbytes = await _drain(stream)
    assert chunks >= 1 and nbytes > 0


async def test_unadvertised_voice_rejected_before_generate(started_backend):
    with pytest.raises((ValueError, KeyError)):
        await started_backend.open_stream(
            voice="hf://gradium/phonon-voices/voices/x.safetensors", language=None, extras=None
        )


async def test_cancel_mid_stream_then_reuse(started_backend):
    for voice in (None, _NAMED):
        stream = await started_backend.open_stream(voice=voice, language=None, extras=None)
        await stream.feed(_SENTENCE + " " + _SENTENCE)
        await stream.end()
        seen = 0
        async for ev in stream.events():
            if ev.kind == "delta" and ev.pcm:
                seen += 1
                if seen == 2:
                    break
        await stream.cancel()
        await asyncio.wait_for(stream.wait_closed(), 10.0)

        # Model still healthy: a fresh stream on the same model completes.
        again = await started_backend.open_stream(voice=None, language=None, extras=None)
        await again.feed("Hello there, this is a follow up.")
        await again.end()
        _chunks, nbytes = await _drain(again)
        assert nbytes > 0


async def test_server_roundtrip_named_voice(monkeypatch):
    # Own backend: running_server closes it on exit, which would break the
    # module-scoped fixture for later tests.
    monkeypatch.setenv(_VOICES_ENV, _NAMED)
    async with (
        running_server(PH.PhononBackend()) as srv,
        connected_client(srv) as (client, _h),
    ):
        resp = await synthesize_once(client, _SENTENCE, voice=_NAMED)
        assert resp.done is not None and resp.failed is None
        assert resp.pcm


async def test_offline_start_after_cache_warm(started_backend):
    """``started_backend`` warmed the cache for the pinned model + enabled voice;
    a fresh process with HF_HUB_OFFLINE=1 must start and advertise the voice."""
    prog = (
        "import asyncio\n"
        "from tts_server.backends.phonon import PhononBackend\n"
        "async def main():\n"
        "    b = PhononBackend()\n"
        "    await b.start()\n"
        "    print(b.sample_rate, b.voices())\n"
        "    await b.close()\n"
        "asyncio.run(main())\n"
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1", _VOICES_ENV: _NAMED}
    result = subprocess.run(
        [sys.executable, "-c", prog],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,  # returncode asserted below
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "24000" in result.stdout and _NAMED in result.stdout


async def test_long_text_seconds_per_word_in_band(started_backend):
    """Q8 recorded ~0.30 s/word with no truncation to 5161 chars. A ~2000-char
    commit (the advertised cap) must stay inside a generous band around that."""
    story = _STORY.read_text(encoding="utf-8")
    limit = PH._MAX_TEXT_CHARS
    cut = story[:limit]
    end = max(cut.rfind(". "), cut.rfind(".\n"))
    text = cut[: end + 1] if end > 0 else cut
    words = len(text.split())
    assert words > 250

    stream = await started_backend.open_stream(voice=None, language=None, extras=None)
    await stream.feed(text)
    await stream.end()
    _chunks, nbytes = await _drain(stream)
    per_word = (nbytes / 2 / 24000) / words
    assert 0.2 <= per_word <= 0.45, f"{per_word:.3f} s/word outside Q8 band (truncation?)"
