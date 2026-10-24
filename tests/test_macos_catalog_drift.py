"""The macOS app's Swift copies of server facts — LEAN drift guard.

``macos/Sources/PipecatToolbarCore/Service.swift`` repeats, per TTS backend, the
``serve --backend`` name (as the package extra), the canonical port and the
default model repo; ``PrefetchEvent.swift`` repeats the supervised error codes.
Swift cannot import Python, so these tests parse the Swift source and assert it
equals the Python/justfile source of truth:

  * the TTS family set equals ``BACKEND_CHOICES``;
  * each family's port equals the justfile ``_resolve`` port;
  * each family's ``modelRepo`` equals the backend's ``DEFAULT_*_MODEL``
    (``tone`` has none);
  * ``SupervisedErrorCode`` raw values equal ``tts_server.supervise``'s ``CODE_*``.
"""

from __future__ import annotations

import re
from pathlib import Path

from tts_server import supervise
from tts_server.__main__ import BACKEND_CHOICES

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "macos" / "Sources" / "PipecatToolbarCore"

_FAMILY = re.compile(
    r'service: \.(\w+), kind: \.tts,.*?runtime: \.ttsServer\(extra: (?:nil|"(\w+)")\),'
    r'\s*port: (\d+),\s*modelRepo: (?:nil|"([^"]+)")',
    re.DOTALL,
)
_RESOLVE = re.compile(r'^\s+(\w+)\)\s+printf .*?"127\.0\.0\.1"\s+"(\d+)"', re.MULTILINE)
_DEFAULT_MODEL = re.compile(r'^DEFAULT_\w+_MODEL = "([^"]+)"', re.MULTILINE)


def _swift_families() -> dict[str, tuple[int, str | None]]:
    text = (CORE / "Service.swift").read_text()
    families = {}
    for case, extra, port, repo in _FAMILY.findall(text):
        name = extra or case  # only ``tone`` has no extra; its case is its name
        families[name] = (int(port), repo or None)
    return families


def test_swift_tts_families_match_backend_choices():
    assert sorted(_swift_families()) == sorted(BACKEND_CHOICES)


def test_swift_ports_match_justfile_resolve():
    resolve = {name: int(port) for name, port in _RESOLVE.findall((REPO / "justfile").read_text())}
    assert sorted(resolve) == sorted(BACKEND_CHOICES)
    assert {name: port for name, (port, _) in _swift_families().items()} == resolve


def test_swift_model_repos_match_backend_defaults():
    for name, (_, repo) in _swift_families().items():
        if name == "tone":
            assert repo is None
            continue
        source = (REPO / "tts_server" / "backends" / f"{name}.py").read_text()
        defaults = _DEFAULT_MODEL.findall(source)
        assert len(defaults) == 1, f"{name}: expected one DEFAULT_*_MODEL"
        assert repo == defaults[0], f"{name}: Swift {repo!r} != Python {defaults[0]!r}"


def test_swift_supervised_error_codes_match_supervise():
    swift = set(
        re.findall(
            r'case \w+ = "(\w+)"',
            (CORE / "PrefetchEvent.swift")
            .read_text()
            .split("enum SupervisedErrorCode", 1)[1]
            .split("}", 1)[0],
        )
    )
    python = {v for k, v in vars(supervise).items() if k.startswith("CODE_")}
    assert swift == python
