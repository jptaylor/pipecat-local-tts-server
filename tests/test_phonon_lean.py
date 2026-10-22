"""Phonon backend LEAN tests (no mlx, no huggingface_hub) — Phase 1.

On the lean allow-list. Everything heavy is reached through the module-level
patchable seams ``_snapshot_download`` / ``_load_model`` / ``_download_voice``,
so ``start()`` (pin logic, voice resolution, soft failure) runs without mlx.

Invariants asserted here:
  * ``phonon.py`` / ``phonon_voices.py`` import with no mlx_audio / numpy /
    huggingface_hub at module load;
  * the pinned model sha reaches ``_snapshot_download`` and ``_load_model``
    receives the LOCAL snapshot path; a download failure never falls back to a
    load of the unpinned repo;
  * the advertised voices are ``default`` + the enabled voices that resolved
    (unknown / unfetchable ones are skipped with a warning, never fatal);
  * no client string, ``ref_audio`` or ``frames_after_eos`` ever reaches
    ``generate()``: ``voice`` is omitted (default) or an absolute existing
    ``.safetensors`` path.

Dual-wire assertions (argparse choice, ``_resolve_model``) are Phase 2.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tts_server.backends import phonon as PH
from tts_server.backends import phonon_voices as PV

from ._helpers import connected_client, running_server

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_SHA = "52433c46975bfa13144c56204c9f4ee1c5b148aa"
_VOICES_SHA = "0ee17205f254b5abc43b266c8a0f215565c352f7"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_LOGGER = "tts_server.backends.phonon"


# --- catalog access (tolerant of the exact container shape) ------------------


def _first_attr(mod, names):
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise AssertionError(f"{mod.__name__} exposes none of {names}")


def _field(v, names):
    if isinstance(v, dict):
        for n in names:
            if n in v:
                return v[n]
    elif isinstance(v, (tuple, list)):
        return v[0] if names[0] in ("voice_id", "id") else v[1]
    else:
        for n in names:
            if hasattr(v, n):
                return getattr(v, n)
    raise AssertionError(f"cannot read {names} from catalog entry {v!r}")


def _catalog() -> list[tuple[str, str, str]]:
    """Normalise the bundled table to ``[(slug, voice_id, language)]``."""
    raw = _first_attr(PV, ["CATALOG", "VOICE_CATALOG", "VOICES"])
    out = []
    if isinstance(raw, dict):
        for slug, v in raw.items():
            out.append((slug, _field(v, ["voice_id", "id"]), _field(v, ["language", "lang"])))
    else:
        for v in raw:
            if isinstance(v, (tuple, list)) and len(v) == 3:  # (slug, voice_id, language)
                out.append(tuple(v))
            else:
                out.append(
                    (
                        _field(v, ["slug", "name"]),
                        _field(v, ["voice_id", "id"]),
                        _field(v, ["language", "lang"]),
                    )
                )
    return out


def _slug_fn():
    return _first_attr(PV, ["slugify", "slug", "make_slug", "voice_slug"])


def _en_slugs(n: int, exclude=("alba",)) -> list[str]:
    return [s for s, _i, lang in _catalog() if lang == "en" and s not in exclude][:n]


# --- fakes / seams ------------------------------------------------------------


class _SpyModel:
    sample_rate = 24000

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return iter(())


class _Miss(Exception):
    """Stands in for LocalEntryNotFoundError / offline failures."""


class Seams:
    """Records seam calls; ``cached`` / ``fail`` control voice resolution."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.snapshot_calls: list[tuple[tuple, dict]] = []
        self.load_calls: list[Any] = []
        self.voice_calls: list[dict] = []  # {id, local_only}
        self.model = _SpyModel()
        self.snapshot_exc: Exception | None = None
        self.cached: set[str] = set()  # voice ids "already in cache"
        self.unfetchable: set[str] = set()  # voice ids that always fail
        self._ids = {i for _s, i, _l in _catalog()}

    def _voice_id(self, args, kwargs) -> str:
        for v in [*args, *kwargs.values()]:
            if not isinstance(v, str):
                continue
            m = re.search(r"voices/(.+?)\.safetensors", v)
            if m:
                return m.group(1)
            if v in self._ids:
                return v
        raise AssertionError(f"_download_voice called without a voice id: {args} {kwargs}")

    def snapshot_download(self, *args, **kwargs):
        self.snapshot_calls.append((args, kwargs))
        if self.snapshot_exc:
            raise self.snapshot_exc
        d = self.tmp / "snapshot"
        d.mkdir(exist_ok=True)
        return str(d)

    def load_model(self, *args, **kwargs):
        self.load_calls.append(args[0] if args else kwargs)
        return self.model

    def download_voice(self, *args, **kwargs):
        vid = self._voice_id(args, kwargs)
        local_only = bool(kwargs.get("local_files_only", False))
        self.voice_calls.append({"id": vid, "local_only": local_only})
        if vid in self.unfetchable:
            raise _Miss(vid)
        if local_only and vid not in self.cached:
            raise _Miss(vid)
        p = self.tmp / "voices" / f"{vid}.safetensors"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    @property
    def network_voice_calls(self) -> list[dict]:
        return [c for c in self.voice_calls if not c["local_only"]]


@pytest.fixture
def seams(monkeypatch, tmp_path):
    monkeypatch.delenv("PIPECAT_TTS_PHONON_VOICES", raising=False)
    s = Seams(tmp_path)
    monkeypatch.setattr(PH, "_snapshot_download", s.snapshot_download)
    monkeypatch.setattr(PH, "_load_model", s.load_model)
    monkeypatch.setattr(PH, "_download_voice", s.download_voice)
    return s


def _sha_in(args, kwargs) -> list[str]:
    return [v for v in [*args, *kwargs.values()] if isinstance(v, str) and _SHA_RE.match(v)]


async def _started(monkeypatch, *, voices: str | None = None, model: str | None = None):
    if voices is not None:
        monkeypatch.setenv("PIPECAT_TTS_PHONON_VOICES", voices)
    backend = PH.PhononBackend(model=model) if model else PH.PhononBackend()
    await backend.start()
    return backend


# --- catalog integrity (phonon_voices) ----------------------------------------


def test_catalog_counts_match_phase0():
    cat = _catalog()
    assert len(cat) == 102
    by_lang: dict[str, int] = {}
    for _s, _i, lang in cat:
        by_lang[lang] = by_lang.get(lang, 0) + 1
    assert by_lang == {"en": 37, "fr": 15, "de": 13, "es": 25, "pt": 12}


def test_catalog_slugs_unique_charset_and_default_reserved():
    slugs = [s for s, _i, _l in _catalog()]
    assert len(set(slugs)) == len(slugs)
    assert all(re.fullmatch(r"[a-z0-9_]+", s) for s in slugs)
    assert "default" not in slugs
    assert "alba" in slugs  # real catalog slug (not the pocket voice)


def test_catalog_voice_ids_unique():
    ids = [i for _s, i, _l in _catalog()]
    assert len(set(ids)) == len(ids)
    assert all(isinstance(i, str) and i for i in ids)


def test_voices_repo_sha_pinned():
    sha = _first_attr(PV, ["VOICES_REPO_SHA", "VOICES_REVISION", "VOICES_SHA"])
    assert sha == _VOICES_SHA


def test_slug_function_rules():
    slug = _slug_fn()
    assert slug("Elodie-Rose") == "elodie_rose"
    assert slug("Lucía-Sol") == "lucia_sol"  # NFKD-strip diacritics
    assert slug("Captain Silas Blackwater") == "captain_silas_blackwater"
    assert slug("Zoey") == "zoey"
    # Plan's literal rule: each non-alphanumeric char -> "_" (no run-collapse).
    assert slug("a--b") == "a__b"


def test_resolve_enabled_splits_known_and_unknown():
    en = _en_slugs(1)[0]
    resolved, unknown = PV.resolve_enabled(["alba", en, "no_such_voice"])
    assert list(unknown) == ["no_such_voice"]
    assert len(resolved) == 2
    assert PV.resolve_enabled([]) == ([], []) or tuple(map(list, PV.resolve_enabled([]))) == (
        [],
        [],
    )


def test_resolve_enabled_never_resolves_default():
    resolved, unknown = PV.resolve_enabled(["default"])
    assert not list(resolved)
    assert list(unknown) == ["default"]


# --- capabilities --------------------------------------------------------------


def test_constants():
    assert PH.DEFAULT_PHONON_MODEL == "gradium/phonon-31m-en"
    assert PH._STREAMING_INTERVAL == 0.3
    assert PH.PhononBackend.backend_name == "phonon"


def test_capabilities_shape_and_negative_guard():
    caps = PH.PhononBackend().capabilities()
    assert caps["streaming"] is True
    assert caps["binary_audio"] is False
    assert caps["text_formats"] == ["plain"]
    assert caps["languages"] == ["en"]
    assert caps["ideal_words"] == 40
    assert caps["max_text_chars"] == 2000
    assert caps["extras"] == ["temperature"]
    assert "ref_audio" not in caps["extras"]
    assert "frames_after_eos" not in caps["extras"]
    assert "streaming_interval" not in caps["extras"]


# --- pin logic via the seams --------------------------------------------------


@pytest.mark.asyncio
async def test_pinned_sha_reaches_download_and_load_gets_local_path(seams, monkeypatch):
    backend = await _started(monkeypatch)
    assert len(seams.snapshot_calls) == 1
    args, kwargs = seams.snapshot_calls[0]
    assert _sha_in(args, kwargs) == [_MODEL_SHA]
    assert any(v == PH.DEFAULT_PHONON_MODEL for v in [*args, *kwargs.values()])
    # load receives the LOCAL snapshot path returned by the download, not the repo id.
    assert seams.load_calls == [str(seams.tmp / "snapshot")]
    assert backend.sample_rate == 24000


@pytest.mark.asyncio
async def test_download_failure_propagates_no_fallback_load(seams, monkeypatch):
    seams.snapshot_exc = RuntimeError("revision not found")
    with pytest.raises(RuntimeError, match="revision not found"):
        await _started(monkeypatch)
    assert seams.load_calls == []  # never loads HEAD / the unpinned repo


@pytest.mark.asyncio
async def test_non_table_model_warns_and_loads_as_given(seams, monkeypatch, caplog, tmp_path):
    snap = str(tmp_path / "my_snapshot")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await _started(monkeypatch, model=snap)
    assert seams.snapshot_calls == []  # no pin applied
    assert seams.load_calls == [snap]
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


# --- voices: env parsing, soft failure, cache ordering -------------------------


@pytest.mark.asyncio
async def test_unset_env_advertises_default_only(seams, monkeypatch):
    backend = await _started(monkeypatch)
    assert backend.voices() == ["default"]
    assert backend.capabilities()["voice_count"] == 1
    assert seams.voice_calls == []


@pytest.mark.asyncio
async def test_enabled_voices_advertised_default_first_dupes_collapsed(seams, monkeypatch):
    en = _en_slugs(1)[0]
    backend = await _started(monkeypatch, voices=f"alba,{en},alba")
    assert backend.voices() == ["default", "alba", en]
    assert backend.capabilities()["voice_count"] == len(backend.voices()) == 3


@pytest.mark.asyncio
async def test_unknown_enabled_name_skipped_with_warning(seams, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        backend = await _started(monkeypatch, voices="alba,no_such_voice")
    assert backend.voices() == ["default", "alba"]
    assert any("no_such_voice" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_unfetchable_voice_skipped_with_warning_server_still_starts(
    seams, monkeypatch, caplog
):
    en = _en_slugs(1)[0]
    en_id = next(i for s, i, _l in _catalog() if s == en)
    seams.unfetchable.add(en_id)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        backend = await _started(monkeypatch, voices=f"alba,{en}")
    assert backend.voices() == ["default", "alba"]
    assert backend.capabilities()["voice_count"] == 2
    assert any(r.levelno >= logging.WARNING and en in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_all_enabled_voices_fail_still_serves_default(seams, monkeypatch):
    seams.unfetchable.update(i for _s, i, _l in _catalog())
    backend = await _started(monkeypatch, voices="alba")
    assert backend.voices() == ["default"]


@pytest.mark.asyncio
async def test_warm_cache_makes_zero_network_voice_calls(seams, monkeypatch):
    seams.cached.update(i for _s, i, _l in _catalog())
    await _started(monkeypatch, voices="alba," + _en_slugs(1)[0])
    assert len(seams.voice_calls) == 2
    assert seams.network_voice_calls == []  # local-files-only tried first


@pytest.mark.asyncio
async def test_cold_cache_falls_back_to_network_per_voice(seams, monkeypatch):
    await _started(monkeypatch, voices="alba")
    modes = [c["local_only"] for c in seams.voice_calls]
    assert modes[0] is True and False in modes  # local first, then network


@pytest.mark.asyncio
async def test_voice_download_carries_voices_sha(seams, monkeypatch):
    seen: list[tuple] = []
    orig = seams.download_voice

    def spy(*a, **k):
        seen.append((a, k))
        return orig(*a, **k)

    monkeypatch.setattr(PH, "_download_voice", spy)
    await _started(monkeypatch, voices="alba")
    assert seen and all(_sha_in(a, k) == [_VOICES_SHA] for a, k in seen)


# --- voice mapping + forbidden kwargs at the backend layer ---------------------


async def _gen_kwargs(backend, *, voice, extras):
    stream = await backend.open_stream(voice=voice, language=None, extras=extras)
    stream._text = "hello world"
    list(stream._gen_factory())
    spy = backend._loaded_model
    assert len(spy.calls) == 1
    return spy.calls[0]


def _reset_calls(backend):
    backend._loaded_model.calls.clear()  # drop the warmup generate


@pytest.mark.asyncio
async def test_default_and_none_omit_voice_kwarg(seams, monkeypatch):
    backend = await _started(monkeypatch)
    for v in (None, "default"):
        _reset_calls(backend)
        call = await _gen_kwargs(backend, voice=v, extras=None)
        assert "voice" not in call
        assert call["stream"] is True
        assert call["streaming_interval"] == PH._STREAMING_INTERVAL


@pytest.mark.asyncio
async def test_enabled_name_forwards_absolute_safetensors_path(seams, monkeypatch):
    backend = await _started(monkeypatch, voices="alba")
    _reset_calls(backend)
    call = await _gen_kwargs(backend, voice="alba", extras=None)
    path = call["voice"]
    assert path != "alba"  # the literal slug never reaches generate()
    assert os.path.isabs(path) and path.endswith(".safetensors")
    assert os.path.isfile(path)


@pytest.mark.asyncio
async def test_not_enabled_alba_rejected_at_backend_layer(seams, monkeypatch):
    # alba IS a catalog slug, but not enabled -> must not be forwarded as a bare name.
    en = _en_slugs(1)[0]
    backend = await _started(monkeypatch, voices=en)
    assert "alba" not in backend.voices()
    with pytest.raises((ValueError, KeyError)):
        await backend.open_stream(voice="alba", language=None, extras=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    ["hf://gradium/phonon-voices/voices/x.safetensors", "nope", "/etc/passwd", "x.safetensors"],
)
async def test_unadvertised_voice_strings_rejected(seams, monkeypatch, bad):
    backend = await _started(monkeypatch, voices="alba")
    with pytest.raises((ValueError, KeyError)):
        await backend.open_stream(voice=bad, language=None, extras=None)


@pytest.mark.asyncio
async def test_every_resolved_voice_argument_is_absolute_existing_path(seams, monkeypatch):
    names = ["alba", *_en_slugs(3)]
    backend = await _started(monkeypatch, voices=",".join(names))
    assert backend.voices() == ["default", *names]
    for name in names:
        _reset_calls(backend)
        call = await _gen_kwargs(backend, voice=name, extras=None)
        assert os.path.isabs(call["voice"]) and call["voice"].endswith(".safetensors")
        assert os.path.isfile(call["voice"])


@pytest.mark.asyncio
async def test_ref_audio_and_frames_after_eos_never_reach_generate(seams, monkeypatch):
    backend = await _started(monkeypatch, voices="alba")
    _reset_calls(backend)
    call = await _gen_kwargs(
        backend,
        voice="alba",
        extras={
            "ref_audio": "clone_me.wav",
            "frames_after_eos": 10,
            "temperature": 0.6,
            "top_k": 50,
            "lsd_decode_steps": 99,
            "noise_clamp": 1.0,
            "eos_threshold": 0.1,
        },
    )
    for forbidden in (
        "ref_audio",
        "frames_after_eos",
        "top_k",
        "lsd_decode_steps",
        "noise_clamp",
        "eos_threshold",
    ):
        assert forbidden not in call
    assert call["temperature"] == 0.6
    assert set(call) <= {"text", "voice", "temperature", "stream", "streaming_interval"}


@pytest.mark.asyncio
async def test_unset_temperature_omitted(seams, monkeypatch):
    backend = await _started(monkeypatch)
    _reset_calls(backend)
    call = await _gen_kwargs(backend, voice=None, extras={})
    assert "temperature" not in call


@pytest.mark.asyncio
async def test_open_stream_before_start_raises(seams):
    with pytest.raises(RuntimeError):
        await PH.PhononBackend().open_stream(voice=None, language=None, extras=None)


def test_validate_extras():
    backend = PH.PhononBackend()
    assert backend.validate_extras({}) is None
    assert backend.validate_extras({"temperature": 0.5}) is None
    msg = backend.validate_extras({"temperature": float("nan")})
    assert msg and "temperature" in msg
    assert backend.validate_extras({"ref_audio": "x"}) is None  # unadvertised: ignored here


# --- server-level: status voices + commit validation ---------------------------


@pytest.mark.asyncio
async def test_server_status_voices_and_rejects_hf_voice(seams, monkeypatch):
    monkeypatch.setenv("PIPECAT_TTS_PHONON_VOICES", "alba")
    backend = PH.PhononBackend()
    async with running_server(backend) as srv, connected_client(srv) as (client, hello):
        assert hello["audio"]["rate"] == 24000
        await client.status()
        status = None
        async for ev in client.events():
            if ev.get("type") == "server.status":
                status = ev
                break
        assert status["voices"][0] == "default"
        assert status["voices"] == ["default", "alba"]
        assert status["voice_count"] == 2

        await client.append("hi")
        await client.commit(voice="hf://x")
        async for ev in client.events():
            if ev.get("type") == "error":
                assert ev["error"]["code"] == "invalid_config"
                break
            assert ev.get("type") != "response.audio.delta"


# --- lazy-import ----------------------------------------------------------------


def _assert_lean(body: str) -> None:
    prog = (
        "import sys\n" + body + "\nheavy = ('mlx_audio', 'mlx', 'numpy', 'huggingface_hub')\n"
        "bad = sorted(n for n in sys.modules if n.split('.')[0] in heavy)\n"
        "print(bad)\n"
        "sys.exit(1) if bad else None\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", prog],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,  # returncode is asserted explicitly below
    )
    assert result.returncode == 0, (
        f"heavy dep leaked or import failed:\n{result.stdout}{result.stderr}"
    )


def test_import_phonon_modules_pulls_no_heavy_dep():
    _assert_lean(
        "import importlib\n"
        "importlib.import_module('tts_server.backends.phonon_voices')\n"
        "importlib.import_module('tts_server.backends.phonon')\n"
    )


def test_make_backend_resolves_phonon_without_heavy_dep():
    _assert_lean(
        "from tts_server.backends import make_backend\n"
        "b = make_backend('phonon')\n"
        "assert b.backend_name == 'phonon', b.backend_name\n"
        "assert b.sample_rate == 0, b.sample_rate\n"
    )


def test_default_model_constant_importable_lean():
    _assert_lean(
        "from tts_server.backends.phonon import DEFAULT_PHONON_MODEL\n"
        "assert DEFAULT_PHONON_MODEL == 'gradium/phonon-31m-en'\n"
    )
