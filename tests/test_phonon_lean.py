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

Dual-wire assertions (argparse choice, ``_resolve_model``) are at the bottom
(Phase 2).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
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
    assert all(PV.SLUG_RE.fullmatch(s) for s in slugs)  # the one slug rule (phonon_voices)
    assert all(PV.is_valid_slug(s) for s in slugs)
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "variant",
    ["Gradium/phonon-31m-en", "gradium/phonon-31m-en/", "  GRADIUM/Phonon-31M-EN// "],
)
async def test_pin_applies_to_normalised_model_variants(seams, monkeypatch, variant):
    await _started(monkeypatch, model=variant)
    assert len(seams.snapshot_calls) == 1
    args, kwargs = seams.snapshot_calls[0]
    assert _sha_in(args, kwargs) == [_MODEL_SHA]
    assert PH.DEFAULT_PHONON_MODEL in [*args, *kwargs.values()]
    assert seams.load_calls == [str(seams.tmp / "snapshot")]


@pytest.mark.asyncio
async def test_unknown_repo_and_similar_local_path_still_unpinned(seams, monkeypatch, tmp_path):
    other = "gradium/phonon-31m-fr"
    await _started(monkeypatch, model=other)
    local = str(tmp_path / "gradium" / "phonon-31m-en")
    await _started(monkeypatch, model=local)
    assert seams.snapshot_calls == []
    assert seams.load_calls == [other, local]


# --- quantized aliases (phonon / phonon-q8 / phonon-q4) -------------------------


class _QuantSpy:
    """Stands in for ``_quantize``: records calls and writes a marker file."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.exc: Exception | None = None

    def __call__(self, snapshot_path, dest, bits):
        self.calls.append((snapshot_path, dest, bits))
        Path(dest).mkdir(parents=True)
        (Path(dest) / "model.safetensors").write_bytes(b"q")
        if self.exc:
            raise self.exc


@pytest.fixture
def quant(seams, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    q = _QuantSpy()
    monkeypatch.setattr(PH, "_quantize", q)
    monkeypatch.setattr(PH, "_mlx_audio_version", lambda: "9.9.9")
    return q


def _qroot(tmp_path: Path) -> Path:
    return tmp_path / "xdg" / "pipecat-tts-server" / "phonon"


@pytest.mark.asyncio
@pytest.mark.parametrize("alias,bits", [("phonon-q8", 8), ("phonon-q4", 4)])
async def test_quant_alias_quantizes_pinned_snapshot_and_loads_that_dir(
    seams, quant, monkeypatch, tmp_path, alias, bits
):
    backend = await _started(monkeypatch, model=alias)
    args, kwargs = seams.snapshot_calls[0]
    assert _sha_in(args, kwargs) == [_MODEL_SHA]
    assert PH.DEFAULT_PHONON_MODEL in [*args, *kwargs.values()]
    snap = str(seams.tmp / "snapshot")
    assert len(quant.calls) == 1
    src, dest, got_bits = quant.calls[0]
    assert (src, got_bits) == (snap, bits)
    final = next(_qroot(tmp_path).iterdir())
    assert final.name == f"{_MODEL_SHA[:12]}-q{bits}-g{PH.Q_GROUP_SIZE}-mlxaudio9.9.9"
    assert dest != str(final) and not Path(dest).exists()  # built in a temp sibling
    assert seams.load_calls == [str(final)]
    assert backend.model == alias


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["PHONON-Q8", "  phonon-q8  ", "Phonon-Q8"])
async def test_quant_alias_case_and_whitespace(seams, quant, monkeypatch, variant):
    backend = await _started(monkeypatch, model=variant)
    assert [c[2] for c in quant.calls] == [8]
    assert backend.model == variant  # reported string is exactly what was passed


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["phonon", "PHONON", PH.DEFAULT_PHONON_MODEL])
async def test_fp_alias_and_repo_id_never_quantize(seams, quant, monkeypatch, tmp_path, model):
    await _started(monkeypatch, model=model)
    assert quant.calls == []
    assert seams.load_calls == [str(seams.tmp / "snapshot")]
    assert not _qroot(tmp_path).exists()
    args, kwargs = seams.snapshot_calls[0]
    assert _sha_in(args, kwargs) == [_MODEL_SHA]


@pytest.mark.asyncio
async def test_existing_quant_dir_is_reused_without_convert(seams, quant, monkeypatch):
    await _started(monkeypatch, model="phonon-q8")
    await _started(monkeypatch, model="phonon-q8")
    assert len(quant.calls) == 1
    assert seams.load_calls[0] == seams.load_calls[1]


@pytest.mark.asyncio
async def test_q8_and_q4_use_distinct_dirs(seams, quant, monkeypatch):
    await _started(monkeypatch, model="phonon-q8")
    await _started(monkeypatch, model="phonon-q4")
    assert seams.load_calls[0] != seams.load_calls[1]
    assert len(quant.calls) == 2


@pytest.mark.asyncio
async def test_quantize_failure_leaves_no_dir_and_raises(seams, quant, monkeypatch, tmp_path):
    quant.exc = RuntimeError("convert boom")
    with pytest.raises(RuntimeError, match="convert boom"):
        await _started(monkeypatch, model="phonon-q8")
    assert seams.load_calls == []  # no silent fp fallback
    root = _qroot(tmp_path)
    assert list(root.iterdir()) == []  # neither the final dir nor a temp leftover
    # A retry after the failure converts again (nothing half-built was reused).
    quant.exc = None
    await _started(monkeypatch, model="phonon-q8")
    assert len(quant.calls) == 2
    assert len(seams.load_calls) == 1


def _plant(root: Path, name: str, age_s: float) -> Path:
    """A dir holding one file, with both mtimes ``age_s`` seconds in the past."""
    d = root / name
    d.mkdir(parents=True)
    f = d / "weights.safetensors"
    f.write_bytes(b"x")
    t = time.time() - age_s
    os.utime(f, (t, t))
    os.utime(d, (t, t))
    return d


@pytest.mark.asyncio
async def test_start_sweeps_only_stale_quantize_temp_dirs(seams, quant, monkeypatch, tmp_path):
    root = _qroot(tmp_path)
    base = PH._quantized_dir(_MODEL_SHA, 8).name
    day = 24 * 3600.0
    stale = _plant(root, f".{base}.tmp-111-222", day)
    fresh = _plant(root, f".{base}.tmp-333-444", 60.0)
    odd_name = _plant(root, f".{base}.tmp-notapid", day)  # not our naming scheme
    other = _plant(root, "keep-me", day)  # a completed dir, however old
    target = _plant(tmp_path / "elsewhere", "payload", day)
    link = root / f".{base}.tmp-555-666"
    link.symlink_to(target, target_is_directory=True)
    plain_file = root / f".{base}.tmp-777-888"
    plain_file.write_bytes(b"x")
    os.utime(plain_file, (time.time() - day,) * 2)

    await _started(monkeypatch, model="phonon-q8")

    assert not stale.exists()
    for kept in (fresh, odd_name, other, target, plain_file):
        assert kept.exists(), kept
    assert link.is_symlink()
    assert len(quant.calls) == 1  # the sweep did not stop the build


def test_sweep_age_boundary_is_ten_minutes(tmp_path):
    assert PH._STALE_TMP_AGE_S == 600.0
    root = tmp_path / "phonon"
    young = _plant(root, ".k.tmp-1-2", 9 * 60.0)
    old = _plant(root, ".k.tmp-3-4", 11 * 60.0)
    PH._sweep_stale_tmp(root)
    assert young.exists()
    assert not old.exists()


def test_sweep_survives_missing_parent_and_unreadable_entries(tmp_path, monkeypatch):
    PH._sweep_stale_tmp(tmp_path / "does-not-exist")  # no cache dir yet: no error
    root = tmp_path / "phonon"
    stale = _plant(root, ".k.tmp-1-2", 24 * 3600.0)

    def boom(*_a, **_k):
        raise OSError("denied")

    monkeypatch.setattr(PH.shutil, "rmtree", boom)
    PH._sweep_stale_tmp(root)  # a failed delete is swallowed, entry left in place
    assert stale.exists()


def test_sweep_skips_an_entry_that_vanishes_mid_scan_and_continues(tmp_path, monkeypatch):
    """An entry another process renames/removes between ``scandir`` and ``stat``
    (a concurrent convert finishing) raises OSError for that entry only."""
    root = tmp_path / "phonon"
    day = 24 * 3600.0
    raced = _plant(root, ".a.tmp-1-2", day)
    other = _plant(root, ".b.tmp-3-4", day)
    real = PH._newest_mtime

    def flaky(path: str) -> float:
        if path.endswith(".a.tmp-1-2"):
            raise FileNotFoundError(path)
        return real(path)

    monkeypatch.setattr(PH, "_newest_mtime", flaky)
    PH._sweep_stale_tmp(root)
    assert raced.exists()  # unreadable entry left alone, not an error
    assert not other.exists()  # and the scan carried on to the next one


def test_quantized_dir_key_tracks_version(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(PH, "_mlx_audio_version", lambda: "1.0")
    a = PH._quantized_dir(_MODEL_SHA, 8)
    monkeypatch.setattr(PH, "_mlx_audio_version", lambda: "2.0")
    assert PH._quantized_dir(_MODEL_SHA, 8) != a
    assert PH._quantized_dir("f" * 40, 8).name.startswith("ffffffffffff-q8-")


def test_cache_root_falls_back_to_home_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert PH._cache_root() == tmp_path / ".cache"


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


@pytest.mark.asyncio
async def test_hostile_and_mixed_case_env_tokens(seams, monkeypatch, caplog):
    """Only catalog slugs resolve (lowercased/stripped); everything else is
    warned + skipped and never reaches the download seam."""
    raw = "hf://x,../x, ALBA ,, ZoEy ,"
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        backend = await _started(monkeypatch, voices=raw)
    assert backend.voices() == ["default", "alba", "zoey"]
    ids = {s: i for s, i, _l in _catalog()}
    assert {c["id"] for c in seams.voice_calls} == {ids["alba"], ids["zoey"]}
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "hf://x" in msgs and "../x" in msgs


@pytest.mark.asyncio
async def test_default_in_env_is_accepted_without_unknown_voice_warning(seams, monkeypatch, caplog):
    """``default`` is always advertised; listing it must not warn "not in catalog"."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        backend = await _started(monkeypatch, voices="default,alba")
    assert backend.voices() == ["default", "alba"]
    assert not [r for r in caplog.records if "default" in r.getMessage()]


@pytest.mark.asyncio
async def test_voice_path_containing_at_sign_is_skipped_at_start(seams, monkeypatch, caplog):
    """mlx-audio's pocket_tts decides "is this a .safetensors file" from the text before
    the first '@', so a voice path with '@' is decoded as audio and fails at synthesis
    (DecodeError). It must be skipped with a warning at start, not advertised."""
    seams.tmp = seams.tmp / "a@b"
    seams.tmp.mkdir()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        backend = await _started(monkeypatch, voices="alba")
    assert backend.voices() == ["default"]
    assert any("'@'" in r.getMessage() and "alba" in r.getMessage() for r in caplog.records)


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
        error = None
        async for ev in client.events():
            if ev.get("type") == "error":
                error = ev
                break
            assert ev.get("type") != "response.audio.delta"
        assert error is not None, "stream ended without an error event"
        assert error["error"]["code"] == "invalid_config"


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


# --- dual-wire (Phase 2): argparse choice + _resolve_model + make_backend ------


def test_phonon_is_accepted_backend_choice():
    """argparse must accept ``--backend phonon`` (not just ``make_backend``)."""
    from tts_server.__main__ import build_parser

    args = build_parser().parse_args(["serve", "--backend", "phonon"])
    assert args.backend == "phonon"


def test_default_phonon_model_resolves_identically_through_both_paths():
    """``_resolve_model`` and ``make_backend`` fallback must agree."""
    from tts_server.__main__ import _resolve_model
    from tts_server.backends import make_backend

    assert _resolve_model("phonon", None) == PH.DEFAULT_PHONON_MODEL
    assert make_backend("phonon").model == PH.DEFAULT_PHONON_MODEL
    # An explicit --model still wins over the default.
    assert _resolve_model("phonon", "x/y") == "x/y"


# --- ~45 s sentence-chunk tripwire ---------------------------------------------


def _chunk_fn(*frame_counts):
    """A fake per-chunk generator function: call N yields ``frame_counts[N]`` frames."""
    it = iter(frame_counts)

    def fn(*_a, **_k):
        yield from range(next(it))

    return fn


def test_tripwire_passes_frames_and_stays_quiet_under_ceiling():
    w = PH._cap_chunk_frames(_chunk_fn(5), 10, PH._make_truncation_error)
    assert list(w()) == [0, 1, 2, 3, 4]


def test_tripwire_yields_every_frame_then_raises_at_ceiling():
    seen = []
    with pytest.raises(PH.PhononTruncationError, match="45 s"):
        for f in PH._cap_chunk_frames(_chunk_fn(10), 10, PH._make_truncation_error)():
            seen.append(f)
    assert seen == list(range(10))  # already-yielded audio stays yielded


def test_tripwire_counts_per_chunk_not_across_chunks():
    w = PH._cap_chunk_frames(_chunk_fn(6, 6, 6), 10, PH._make_truncation_error)
    for _ in range(3):  # 18 frames in total, but never 10 in one chunk
        assert len(list(w())) == 6


def test_tripwire_early_close_never_raises():
    gen = PH._cap_chunk_frames(_chunk_fn(100), 10, PH._make_truncation_error)()
    assert next(gen) == 0
    gen.close()  # barge-in: no exception, though the chunk would have hit the cap


def test_install_wraps_once_and_reports_missing_method():
    class M:
        calls = 0

        def _generate_audio_stream_short_text(self, **_k):
            M.calls += 1
            yield from range(PH._CHUNK_FRAMES_CEILING)

    m = M()
    assert PH._install_chunk_tripwire(m, PH._CHUNK_FRAMES_CEILING) is True
    wrapped = m._generate_audio_stream_short_text
    assert PH._install_chunk_tripwire(m, PH._CHUNK_FRAMES_CEILING) is True
    assert m._generate_audio_stream_short_text is wrapped  # idempotent, not double-wrapped
    with pytest.raises(PH.PhononTruncationError):
        list(m._generate_audio_stream_short_text(text_to_generate="x"))
    assert PH._install_chunk_tripwire(object(), PH._CHUNK_FRAMES_CEILING) is False


async def test_start_warns_when_model_lacks_the_chunk_method(seams, caplog, monkeypatch):
    with caplog.at_level("WARNING"):
        await _started(monkeypatch)
    assert any("truncation guard is OFF" in r.getMessage() for r in caplog.records)


class _ChunkSpyModel(_SpyModel):
    """A model that HAS the private per-chunk method, yielding exactly the ceiling."""

    def _generate_audio_stream_short_text(self, **_k):
        yield from range(PH._CHUNK_FRAMES_CEILING)


@pytest.mark.asyncio
@pytest.mark.parametrize("has_method", [True, False])
@pytest.mark.parametrize(
    ("model", "pinned"),
    [
        (None, True),  # the default repo id
        ("phonon", True),
        ("phonon-q8", True),
        ("phonon-q4", True),
        ("  Gradium/phonon-31m-en/ ", True),
        ("gradium/phonon-31m-fr", False),  # unpinned repo
        ("LOCAL", False),  # absolute local snapshot path
    ],
)
async def test_chunk_tripwire_installed_iff_pinned_and_method_present(
    seams, quant, monkeypatch, caplog, tmp_path, model, pinned, has_method
):
    """Regression (gauntlet r2 D): the 560-frame / 12.5 Hz ceiling is measured only on
    the pinned model, so an unpinned repo or local path must NOT get the tripwire even
    when it exposes ``_generate_audio_stream_short_text`` (before the fix it was
    wrapped unconditionally). Whole state space: pinned x method present; the guard is
    on only in the (pinned, present) cell and every other cell warns "guard is OFF"."""
    seams.model = _ChunkSpyModel() if has_method else _SpyModel()
    if model == "LOCAL":
        model = str(tmp_path / "gradium" / "phonon-31m-en")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        backend = await _started(monkeypatch, model=model)
    m = backend._loaded_model
    fn = getattr(m, "_generate_audio_stream_short_text", None)
    wrapped = bool(getattr(fn, "_phonon_tripwire", False))
    assert wrapped is (pinned and has_method)
    if has_method:
        if wrapped:
            with pytest.raises(PH.PhononTruncationError):
                list(fn(text_to_generate="x"))
        else:  # unwrapped: the same ceiling-length chunk passes through untouched
            assert len(list(fn(text_to_generate="x"))) == PH._CHUNK_FRAMES_CEILING
    off = [r.getMessage() for r in caplog.records if "truncation guard is OFF" in r.getMessage()]
    assert bool(off) is (not wrapped)
    if not pinned:
        assert "no revision pin" in off[0]


@pytest.mark.asyncio
async def test_chunk_tripwire_off_for_pinned_repo_without_measured_ceiling(
    seams, monkeypatch, caplog
):
    """Regression (gauntlet r3 A): the tripwire gate must read the per-repo measured
    ceiling table, not the loader-compat ``_REVISION_PINS``. Before the fix, pinning a
    second repo (for loader reasons) switched the tripwire on with the default model's
    unmeasured ceiling."""
    other = "gradium/phonon-31m-fr"
    monkeypatch.setitem(PH._REVISION_PINS, other, "a" * 40)
    seams.model = _ChunkSpyModel()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        backend = await _started(monkeypatch, model=other)
    fn = backend._loaded_model._generate_audio_stream_short_text
    assert not getattr(fn, "_phonon_tripwire", False)
    assert len(list(fn(text_to_generate="x"))) == PH._CHUNK_FRAMES_CEILING
    off = [r.getMessage() for r in caplog.records if "truncation guard is OFF" in r.getMessage()]
    assert off and "no measured chunk-frame ceiling" in off[0]


@pytest.mark.asyncio
async def test_chunk_tripwire_off_when_default_pin_sha_changes(seams, monkeypatch, caplog):
    """Regression (gauntlet r4 A): a ceiling is bound to the exact ``(repo, sha)`` it was
    measured on. Before the fix the table was keyed by repo only, so bumping the default
    repo's pin sha without re-measuring kept installing the old revision's 560."""
    new_sha = "b" * 40
    monkeypatch.setitem(PH._REVISION_PINS, PH.DEFAULT_PHONON_MODEL, new_sha)
    seams.model = _ChunkSpyModel()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        backend = await _started(monkeypatch)
    fn = backend._loaded_model._generate_audio_stream_short_text
    assert not getattr(fn, "_phonon_tripwire", False)
    assert len(list(fn(text_to_generate="x"))) == PH._CHUNK_FRAMES_CEILING
    off = [r.getMessage() for r in caplog.records if "truncation guard is OFF" in r.getMessage()]
    assert off and "no measured chunk-frame ceiling" in off[0]
    assert f"{PH.DEFAULT_PHONON_MODEL}@{new_sha}" in off[0]


class _NFrameChunkModel(_SpyModel):
    def __init__(self, frames):
        super().__init__()
        self.frames = frames

    def _generate_audio_stream_short_text(self, **_k):
        yield from range(self.frames)


@pytest.mark.asyncio
@pytest.mark.parametrize(("frames", "raises"), [(9, False), (10, True)])
async def test_start_forwards_the_per_revision_ceiling(seams, monkeypatch, frames, raises):
    """Regression (gauntlet r4 C): ``start()`` must pass the table's ceiling for the
    resolved pin, not the module default. With the entry patched to 10, a 10-frame chunk
    must trip and a 9-frame chunk must pass (the default 560 would pass both)."""
    key = (PH.DEFAULT_PHONON_MODEL, PH._REVISION_PINS[PH.DEFAULT_PHONON_MODEL])
    monkeypatch.setitem(PH._CHUNK_FRAMES_CEILINGS, key, 10)
    seams.model = _NFrameChunkModel(frames)
    backend = await _started(monkeypatch)
    fn = backend._loaded_model._generate_audio_stream_short_text
    assert getattr(fn, "_phonon_tripwire", False)
    if raises:
        with pytest.raises(PH.PhononTruncationError):
            list(fn(text_to_generate="x"))
    else:
        assert len(list(fn(text_to_generate="x"))) == frames


def test_every_measured_ceiling_is_for_its_exact_pinned_revision():
    """A ceiling is a model fact measured at a specific pinned revision: every
    ``(repo, sha)`` key must name a pinned repo pinned at that SAME sha."""
    for repo, sha in PH._CHUNK_FRAMES_CEILINGS:
        assert PH._REVISION_PINS.get(repo) == sha, (repo, sha)


# --- voice path preconditions (_resolve_voices) ---------------------------------


@pytest.mark.parametrize(
    ("kind", "kept"),
    [
        ("ok", True),
        ("relative", False),
        ("missing", False),
        ("directory", False),
        ("at_sign", False),
        ("no_suffix", False),  # e.g. /etc/passwd
        ("wav", False),
        ("suffix_not_last", False),
        ("upper_suffix", False),  # mlx-audio's suffix check is case-sensitive
    ],
)
def test_resolve_voices_path_preconditions_whole_state_space(
    monkeypatch, tmp_path, caplog, kind, kept
):
    """Regression (gauntlet r2 E): a resolved voice path is kept only if it is
    absolute, an existing file, '@'-free AND ends in ``.safetensors``; before the fix
    a non-``.safetensors`` file (the ``no_suffix``/``wav``/... cells) was stored and
    later forwarded to ``generate(voice=...)``."""
    paths = {
        "ok": tmp_path / "v" / "x.safetensors",
        "relative": Path("v") / "x.safetensors",
        "missing": tmp_path / "nope.safetensors",
        "directory": tmp_path / "d.safetensors",
        "at_sign": tmp_path / "a@b" / "x.safetensors",
        "no_suffix": tmp_path / "passwd",
        "wav": tmp_path / "x.wav",
        "suffix_not_last": tmp_path / "x.safetensors.bak",
        "upper_suffix": tmp_path / "x.SAFETENSORS",
    }
    path = paths[kind]
    if kind == "directory":
        path.mkdir()
    elif kind not in ("relative", "missing"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    monkeypatch.setattr(PH, "_download_voice", lambda *_a, **_k: str(path))
    backend = PH.PhononBackend(enabled_voices=["zoey"])
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        got = backend._resolve_voices()
    assert got == ({"default": None, "zoey": str(path)} if kept else {"default": None})
    skipped = [
        r for r in caplog.records if "zoey" in r.getMessage() and "skipped" in r.getMessage()
    ]
    assert bool(skipped) is (not kept)


@pytest.mark.asyncio
async def test_non_safetensors_voice_path_never_reaches_generate(seams, monkeypatch, caplog):
    """End to end: a download seam returning a real non-voice file (``passwd``) is
    skipped at start, never advertised, and never forwarded to ``generate()``."""
    bogus = seams.tmp / "passwd"
    bogus.write_text("root:x:0:0")
    monkeypatch.setattr(PH, "_download_voice", lambda *_a, **_k: str(bogus))
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        backend = await _started(monkeypatch, voices="zoey")
    assert backend.voices() == ["default"]
    assert str(bogus) not in backend._voice_paths.values()
    with pytest.raises(ValueError):
        await backend.open_stream(voice="zoey", language=None, extras=None)
    assert all(c.get("voice") != str(bogus) for c in backend._loaded_model.calls)
    assert any(".safetensors" in r.getMessage() for r in caplog.records)
