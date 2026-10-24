"""One parser for the macOS app's service catalog (``Service.swift``).

The drift tests (``test_macos_catalog_drift.py``, ``test_justfile_recipes.py``)
compare the Swift copies of server facts against their Python/justfile sources;
they all read the catalog through ``macos_catalog()`` so a variant entry
(``info(for: .phonon).variant(.phononQ8, ..., port: 9366, modelArgument: ...)``)
is seen exactly like a family literal (``ServiceInfo(service: .phonon, ...)``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

SERVICE_SWIFT = (
    Path(__file__).resolve().parents[1]
    / "macos"
    / "Sources"
    / "PipecatToolbarCore"
    / "Service.swift"
)

_CASE = re.compile(r'^\s*case (\w+)(?: = "(\w+)")?\s*$', re.MULTILINE)
_BLOCK = re.compile(r"^ {8}case \.(\w+):\n(.*?)(?=^ {8}case \.|\Z)", re.MULTILINE | re.DOTALL)
_VARIANT = re.compile(r"info\(for: \.(\w+)\)\.variant\(\s*\.(\w+),")


@dataclass(frozen=True)
class CatalogEntry:
    """One ``Service`` case as the app's catalog describes it."""

    name: str  # the raw value (``phonon_q8``): a server backend name for TTS families
    family: str  # the family default's raw value
    kind: str  # "tts" | "stt"
    extra: str | None  # ``runtime: .ttsServer(extra:)`` (TTS only)
    port: int
    model_repo: str | None
    model_argument: str | None
    streaming: bool
    # ``runtime: .sttHelper(engine:requirements:)`` (STT helper engines only)
    engine: str | None = None
    requirements: tuple[str, ...] = ()

    @property
    def is_variant(self) -> bool:
        return self.name != self.family


def _field(block: str, pattern: str) -> str | None:
    m = re.search(pattern, block)
    return m.group(1) if m else None


def _optional_string(block: str, label: str) -> str | None:
    m = re.search(rf'{label}: (?:nil|"([^"]+)")', block)
    return m.group(1) if m else None


def _stt_helper(block: str, constants: dict[str, str]) -> tuple[str | None, tuple[str, ...]]:
    """``(engine, requirements)`` of ``.sttHelper(engine: "x", requirements: [...])``;
    a bare identifier in the list is a ``static let`` string of ``ServiceCatalog``."""
    m = re.search(r'\.sttHelper\(\s*engine: "(\w+)", requirements: \[(.*?)\]\)', block, re.DOTALL)
    if not m:
        return None, ()
    items = []
    for item in (part.strip() for part in m.group(2).split(",")):
        if item.startswith('"'):
            items.append(item.strip('"'))
        elif item:
            assert item in constants, f"unknown requirement constant {item}"
            items.append(constants[item])
    return m.group(1), tuple(items)


def macos_catalog() -> dict[str, CatalogEntry]:
    """``{raw name: entry}`` for every ``Service`` case, variants included."""
    text = SERVICE_SWIFT.read_text()
    constants = dict(re.findall(r'public static let (\w+) = "([^"]+)"', text))
    enum = text.split("public enum Service:", 1)[1].split("public var id", 1)[0]
    raw = {case: value or case for case, value in _CASE.findall(enum)}
    body = text.split("public static func info(for service: Service) -> ServiceInfo {", 1)[1]

    literals: dict[str, CatalogEntry] = {}
    variants: list[tuple[str, str, str]] = []
    for case, block in _BLOCK.findall(body):
        if m := _VARIANT.search(block):
            base, variant = m.groups()
            assert variant == case, f"{case}: builds the variant .{variant}"
            variants.append((case, base, block))
            continue
        assert "ServiceInfo(" in block, f"{case}: neither a ServiceInfo literal nor a variant"
        assert _field(block, r"service: \.(\w+),") == case, f"{case}: literal names another service"
        engine, requirements = _stt_helper(block, constants)
        literals[case] = CatalogEntry(
            engine=engine,
            requirements=requirements,
            name=raw[case],
            family=raw[case],
            kind=_field(block, r"kind: \.(tts|stt),"),
            extra=_optional_string(block, r"runtime: \.ttsServer\(extra"),
            port=int(_field(block, r"port: (\d+)")),
            model_repo=_optional_string(block, "modelRepo"),
            model_argument=_optional_string(block, "modelArgument"),
            streaming=_field(block, r"streaming: (true|false)") == "true",
        )
    entries = dict(literals)
    for case, base, block in variants:
        # A variant shares its family's runtime, kind and streaming flag.
        entries[case] = replace(
            literals[base],
            name=raw[case],
            port=int(_field(block, r"port: (\d+)")),
            model_repo=_optional_string(block, "modelRepo") or literals[base].model_repo,
            model_argument=_optional_string(block, "modelArgument"),
        )
    assert sorted(entries) == sorted(raw), "every Service case has a catalog entry"
    return {entry.name: entry for entry in entries.values()}
