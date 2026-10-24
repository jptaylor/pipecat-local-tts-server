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

import json
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


TRY_IT_FIXTURE = (
    REPO / "macos" / "Tests" / "PipecatServiceToolbarTests" / "Fixtures" / "tts_wire_frames.json"
)
# Per-frame values the server mints fresh on every run.
_VOLATILE = {"event_id", "response_id"}


async def _try_it_exchange() -> dict[str, dict]:
    """The frames the real server sends for the exchange the app's Try it runs:
    each request is an ``input_text.append`` (event id ``<id>.append``) and an
    ``input_text.commit`` (event id ``<id>``), then Stop sends ``response.cancel``."""
    from tts_server.backend import ToneBackend

    from ._helpers import connected_client, next_event, running_server

    async def send(client, frame: dict) -> None:
        await client._ws.send(json.dumps(frame))

    frames: dict[str, dict] = {}
    async with running_server(ToneBackend()) as srv, connected_client(srv) as (client, _hello):
        await send(
            client, {"type": "input_text.append", "text": "Hello", "event_id": "tryit-1.append"}
        )
        await send(client, {"type": "input_text.commit", "event_id": "tryit-1"})
        frames["committed"] = await next_event(client, "input_text.committed")
        frames["created"] = await next_event(client, "response.created")
        # A second request while the first is in flight is refused (K=1).
        await send(
            client, {"type": "input_text.append", "text": "Again", "event_id": "tryit-2.append"}
        )
        await send(client, {"type": "input_text.commit", "event_id": "tryit-2"})
        frames["error_busy"] = await next_event(client, "error")
        await send(client, {"type": "input_text.append", "event_id": "tryit-3.append"})
        frames["error_append"] = await next_event(client, "error")
        frames["audio"] = await next_event(client, "response.audio.delta")
        await send(client, {"type": "response.cancel"})
        frames["cancelled"] = await next_event(client, "response.cancelled")
    frames["audio"]["audio"] = "AID/fw=="  # four pcm16 bytes: -32768 then 32767
    frames["audio"]["seq"] = 0
    return frames


def _shape(value, key=None):
    """``value`` with the volatile ids reduced to their type."""
    if isinstance(value, dict):
        return {k: _shape(v, k) for k, v in value.items()}
    if key in _VOLATILE:
        return type(value).__name__
    return value


async def test_swift_try_it_fixture_is_what_the_server_sends():
    """``tts_wire_frames.json`` (decoded by the Swift ``TTSEvent`` tests) holds
    the real server frames, so the app's decoder is tested against the wire
    format ``server._error`` / ``_error_object`` actually write."""
    fixture = json.loads(TRY_IT_FIXTURE.read_text())
    live = await _try_it_exchange()
    assert sorted(fixture) == sorted(live)
    for name, frame in live.items():
        assert _shape(fixture[name]) == _shape(frame), name
    # The Swift tests rely on these correlations holding in the fixture itself.
    assert fixture["committed"]["previous_event_id"] == "tryit-1"
    assert fixture["committed"]["response_id"] == fixture["created"]["response_id"]
    assert fixture["error_busy"]["previous_event_id"] == "tryit-2"
    assert fixture["error_append"]["previous_event_id"] == "tryit-3.append"


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
