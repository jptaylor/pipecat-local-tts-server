"""``python -m tts_server voices`` (phonon catalog) — LEAN tests.

Isolated: ``HF_HUB_CACHE`` points at ``tmp_path`` and the download seam
``phonon._download_voice`` is stubbed to a fake cache / fake ``voices.jsonl``,
so nothing touches the network or the real HF cache. Invariants:

  * no ``--backend`` defaults to phonon; any other backend exits 2;
  * the ``cached`` column flips when ``voices/<id>.safetensors`` appears;
  * ``--refresh`` only diffs: bundled catalog and cache stay byte-identical,
    and a fetch failure exits 1.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from tts_server import __main__ as M
from tts_server.backends import phonon as PH
from tts_server.backends import phonon_voices as PV


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class _Calls(list):
    """Recorded download calls; ``cache_dirs`` holds the per-call ``cache_dir`` arg."""

    cache_dirs: list


@pytest.fixture
def cache(tmp_path, monkeypatch):
    c = tmp_path / "hf"
    c.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(c))
    monkeypatch.delenv(PH.VOICES_ENV, raising=False)
    calls = _Calls()
    cache_dirs: list = []
    live_rows: list[dict] = []

    def fake(repo, filename, revision, *, local_files_only, cache_dir=None):
        calls.append((repo, filename, revision, local_files_only))
        cache_dirs.append(cache_dir)
        if filename == "voices.jsonl":
            if local_files_only:
                raise FileNotFoundError(filename)
            if live_rows is None:
                raise OSError("network down")
            d = Path(cache_dir) if cache_dir else c
            f = d / "voices.jsonl"
            f.write_text("\n".join(json.dumps(r) for r in live_rows) + "\n")
            return str(f)
        f = c / filename
        if not f.exists():
            raise FileNotFoundError(filename)
        return str(f)

    monkeypatch.setattr(PH, "_download_voice", fake)
    calls.cache_dirs = cache_dirs
    return c, calls, live_rows


def _run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["tts_server", "voices", *argv])
    code = 0
    try:
        M.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    out = capsys.readouterr()
    return code, out.out, out.err


def test_default_backend_is_phonon_and_lists_catalog(cache, monkeypatch, capsys):
    code, out, _ = _run(monkeypatch, capsys, "--json")
    assert code == 0
    rows = json.loads(out)
    assert [r["slug"] for r in rows] == [s for s, _i, _l in PV.VOICE_CATALOG]
    assert all(set(r) == {"slug", "language", "enabled", "cached"} for r in rows)
    assert not any(r["cached"] or r["enabled"] for r in rows)


def test_cached_probe_uses_pinned_revision_and_local_only(cache, monkeypatch, capsys):
    _c, calls, _ = cache
    _run(monkeypatch, capsys, "--json")
    assert calls and all(
        c[0] == PV.VOICES_REPO and c[2] == PV.VOICES_REPO_SHA and c[3] is True for c in calls
    )


def test_cached_column_flips_when_voice_file_present(cache, monkeypatch, capsys):
    c, _calls, _ = cache
    slug = PV.VOICE_CATALOG[0][0]
    f = c / PV.voice_filename(slug)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"fake")
    _, out, _ = _run(monkeypatch, capsys, "--json")
    by_slug = {r["slug"]: r for r in json.loads(out)}
    assert by_slug[slug]["cached"] is True
    assert sum(r["cached"] for r in by_slug.values()) == 1


def test_cached_unknown_when_huggingface_hub_missing(cache, monkeypatch, capsys):
    def no_hub(*a, **k):
        raise ImportError("No module named 'huggingface_hub'")

    monkeypatch.setattr(PH, "_download_voice", no_hub)
    _, out, _ = _run(monkeypatch, capsys, "--json")
    assert all(r["cached"] is None for r in json.loads(out))
    _, out, _ = _run(monkeypatch, capsys)
    assert all(line.split()[-1] == "?" for line in out.splitlines()[1:])


def test_enabled_column_follows_env(cache, monkeypatch, capsys):
    slug = PV.VOICE_CATALOG[0][0]
    monkeypatch.setenv(PH.VOICES_ENV, slug)
    _, out, _ = _run(monkeypatch, capsys, "--json")
    by_slug = {r["slug"]: r for r in json.loads(out)}
    assert by_slug[slug]["enabled"] is True
    assert sum(r["enabled"] for r in by_slug.values()) == 1


def test_language_filter(cache, monkeypatch, capsys):
    lang = PV.VOICE_CATALOG[0][2]
    _, out, _ = _run(monkeypatch, capsys, "--json", "--language", lang)
    rows = json.loads(out)
    assert rows and all(r["language"] == lang for r in rows)
    _, out, _ = _run(monkeypatch, capsys, "--json", "--language", "zz")
    assert json.loads(out) == []


def test_text_output_has_header_and_rows(cache, monkeypatch, capsys):
    code, out, _ = _run(monkeypatch, capsys)
    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == ["slug", "lang", "enabled", "cached"]
    assert len(lines) == 1 + len(PV.VOICE_CATALOG)


@pytest.mark.parametrize("backend", ["kokoro", "tone", "bogus"])
def test_non_phonon_backend_exits_2(cache, monkeypatch, capsys, backend):
    code, out, err = _run(monkeypatch, capsys, "--backend", backend)
    assert code == 2
    assert "no voice catalog for backend" in err
    assert out == ""


def test_refresh_reports_diff_and_modifies_nothing(cache, monkeypatch, capsys):
    c, calls, live = cache
    bundled = list(PV.VOICE_CATALOG)
    # live == bundled minus the first voice, plus one new, with one changed language.
    removed = bundled[0]
    changed = bundled[1]
    live.extend(
        {"name": s, "voice_id": i, "language": ("xx" if s == changed[0] else lang)}
        for s, i, lang in bundled[1:]
    )
    live.append({"name": "Brand New", "voice_id": "vid-new", "language": "en"})
    catalog_before = PV.VOICE_CATALOG
    catalog_file = Path(PV.__file__)
    src_before = catalog_file.read_bytes()
    # Pre-populate the cache so a stray write would show up in the digest.
    (c / "voices").mkdir()
    (c / "voices" / "x.safetensors").write_bytes(b"x")
    before = _tree_digest(c)

    code, out, _ = _run(monkeypatch, capsys, "--refresh")
    assert code == 0
    assert "1 added, 1 removed, 1 changed" in out
    assert f"added: {PV.slugify('Brand New')}" in out
    assert f"removed: {removed[0]}" in out
    assert f"changed: {changed[0]}" in out
    # The refresh fetch: voices.jsonl at ``main``, network allowed.
    assert (PV.VOICES_REPO, "voices.jsonl", "main", False) in calls
    assert PV.VOICE_CATALOG is catalog_before and PV.VOICE_CATALOG == tuple(bundled)
    assert catalog_file.read_bytes() == src_before
    # The live fetch went to a throwaway dir, never the shared cache: the whole
    # cache tree (no exemptions) is byte-identical.
    refresh_dirs = [d for d in calls.cache_dirs if d is not None]
    assert refresh_dirs and all(Path(d) != c and c not in Path(d).parents for d in refresh_dirs)
    assert _tree_digest(c) == before


def test_refresh_identical_catalog_reports_zero(cache, monkeypatch, capsys):
    _c, _calls, live = cache
    live.extend({"name": s, "voice_id": i, "language": lang} for s, i, lang in PV.VOICE_CATALOG)
    code, out, _ = _run(monkeypatch, capsys, "--refresh")
    assert code == 0
    assert "0 added, 0 removed, 0 changed" in out


def test_refresh_fetch_failure_exits_1(cache, monkeypatch, capsys):
    def boom(*a, **k):
        if k.get("local_files_only"):
            raise FileNotFoundError
        raise OSError("network down")

    monkeypatch.setattr(PH, "_download_voice", boom)
    code, _out, err = _run(monkeypatch, capsys, "--refresh")
    assert code == 1
    assert "--refresh failed" in err and "network down" in err
