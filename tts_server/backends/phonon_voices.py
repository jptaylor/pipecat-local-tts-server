"""Bundled Phonon voice catalog (lean: stdlib only, no mlx / huggingface_hub).

A static table derived from ``voices.jsonl`` of ``gradium/phonon-voices`` at the
recorded revision ``VOICES_REPO_SHA`` (102 voices: en 37 / fr 15 / de 13 / es 25 /
pt 12). Each row is ``(slug, voice_id, language)``; the voice file lives at
``voices/<voice_id>.safetensors`` in the voices repo. Slugs are the display name
NFKD-stripped of diacritics, lowercased, each non-alphanumeric character -> ``_``
(unique after slugging; asserted by the catalog tests). ``default`` is reserved
for the model's own shipped voice and is never a catalog slug. The slug rule
(``SLUG_RE`` + the reserved name) is defined once here as ``is_valid_slug``.

The catalog is separate from the operator's *enabled set*
(``PIPECAT_TTS_PHONON_VOICES``): only enabled voices are advertised/downloaded.
"""

from __future__ import annotations

import re
import unicodedata

VOICES_REPO = "gradium/phonon-voices"
VOICES_REPO_SHA = "0ee17205f254b5abc43b266c8a0f215565c352f7"
DEFAULT_VOICE = "default"

# (slug, voice_id, language) — sorted by slug.
VOICE_CATALOG: tuple[tuple[str, str, str], ...] = (
    ("alba", "u2UscyAnHilpiwmf", "es"),
    ("aldo", "ax7dW1XB0eiMfHgO", "es"),
    ("alfie", "X6NceqF285Gr04gb", "en"),
    ("alicia", "4701YgRXjkBcg6sL", "es"),
    ("annika", "p6Uutkyi3j2iNAUu", "de"),
    ("anton", "3ZKKapPOvuWFcw9f", "de"),
    ("arnaud", "aXMVwiUhITMrwPzd", "fr"),
    ("augustin", "Tek4tJXiX6_yvXq7", "fr"),
    ("azgorath", "dGyGDKo2osj7h3s7", "en"),
    ("azucena", "KJpIQlbYrZPA4aLq", "es"),
    ("baptiste", "rqqeErpASHVQXHDd", "fr"),
    ("borja", "tIjCgP8v3RY8UpgM", "es"),
    ("bryony", "gfP44RlHwr2kEwk7", "en"),
    ("candela", "c9wqBDmQiBie5q6Y", "es"),
    ("captain_silas_blackwater", "xPDpsV3F592srkjA", "en"),
    ("celia", "rmLBOdHrfkFym9MM", "es"),
    ("clarice", "sGCMzFIA53ppIngM", "pt"),
    ("coralie", "ZeSg853xFACESHHI", "fr"),
    ("danae", "evfxKSxpDIhmsNoq", "es"),
    ("davi", "NuUr_x5V90hSHzCJ", "pt"),
    ("debora", "K8mj96kYHMy8q3Hb", "pt"),
    ("delia", "nGzMHMPAWGaJFHOo", "en"),
    ("easton", "LMDIYE0Oa377Qh6Z", "en"),
    ("edwin", "zUCxzrOrugTSis03", "en"),
    ("elodie_rose", "y9bQqIDnyxwqK01t", "en"),
    ("everly", "vjFyp2BcCyTlUmYb", "en"),
    ("fairy_godmother", "QXNr0SvkukLB4wPq", "en"),
    ("fennick_the_young_elf", "cXoIXJ2llS8S0WOW", "en"),
    ("flavia", "gOc90Rjmv5CobHCj", "pt"),
    ("freya", "GgfEkEJtxZR7gnpy", "en"),
    ("grub_the_ogre", "cYtsRSxV5q9IMehn", "en"),
    ("guillermo", "FaLVVqpFQcDOtq45", "es"),
    ("hadley", "bXwrkHhURL1VFJdG", "en"),
    ("harper", "4SZHfMpw-p46Ywgs", "en"),
    ("helene", "Vvakv7iWZCUrBooR", "fr"),
    ("henrik", "yyS1KYWs6mXoEw7D", "de"),
    ("hugh", "YyKcxPG7iIgS3jsK", "en"),
    ("igor", "l7520SIXi14iEamd", "pt"),
    ("iker", "t-_TS1e-0GzDAX02", "es"),
    ("imogen", "gDT1nz7Ie36ZhL-C", "en"),
    ("isadora", "CtNvKUUMrx1h56ih", "pt"),
    ("izan", "Ejb29qHnSOK0cIHY", "es"),
    ("jette", "4Mn9VfG2wsLLEzi5", "de"),
    ("jimena", "qG5BXOJYrKEnXLIX", "es"),
    ("josefine", "yadpmqsHVrYC0NJp", "de"),
    ("jule", "MxhgvuHuUER7qjmL", "de"),
    ("jules", "YKeBw3OV1RgpdhLh", "fr"),
    ("juniper", "NIWIOv5vTi5eRMj8", "en"),
    ("kaua", "6HftnVU1XqVBJS8t", "pt"),
    ("lalo", "F05gqkzX7WhKVpSL", "es"),
    ("leon", "20zdyYrQPzKlCwkk", "de"),
    ("leonel", "BkuASiOJRrJEB5qA", "es"),
    ("lilou", "4iKxBtHBRzr1YOph", "fr"),
    ("lotta", "MOrDPn052KUlnKXk", "de"),
    ("lottie", "m9MxpY6IUGXtWp76", "en"),
    ("lucia_sol", "3p0eIGbkmny71GlA", "es"),
    ("lucien", "LIv4NaG8ay4Mhqtd", "fr"),
    ("maelys", "s048cR1l2Jmu4k3B", "fr"),
    ("maeve", "6PWnV0Nq4wu7RVBT", "en"),
    ("manuela_lu", "KgC2Nqnjj48NUiyV", "pt"),
    ("marcelo", "qDeAnE4L4bk0BRz1", "pt"),
    ("marcos", "jvPx8j8zLGQ3utZz", "es"),
    ("marius", "biuhvu17TxVKOcyy", "fr"),
    ("master_aldric", "gSgG5Tor0GPEnqzC", "en"),
    ("mats", "Kf5m22mROozoMWj3", "de"),
    ("micah", "40QrMVgAxFiUnP9f", "en"),
    ("mika", "R3uU1GTAZJotUhUs", "de"),
    ("miles", "d9XCyhjo4JhkHF1d", "en"),
    ("mirko", "Xn9xDzqSSr4xGTNg", "de"),
    ("morwenna_of_the_deep", "nOykTmcH9ydsGGIc", "en"),
    ("nele", "2A2fFf8SkDEJ8nzx", "de"),
    ("nerea", "EQcQFGkYfMn5YJgx", "es"),
    ("nico", "PwgPEaweQILZdGc4", "es"),
    ("ninon", "qtPlUYAZ4EJDuWMm", "fr"),
    ("noa", "b6FvJAiokjdqIti4", "es"),
    ("noemie", "FXxJ9mANRq6BCTX5", "fr"),
    ("nuno", "fghJGlv90qu8OJ36", "es"),
    ("old_mortlake", "lX3LavMNdlIbgEHH", "en"),
    ("orianne", "TUk4X11S6UytB3G6", "fr"),
    ("otavio", "JYtmeaysoaR0Wybf", "pt"),
    ("pippa", "uem82D50GRv2Dwma", "en"),
    ("preston", "qIxlZ7Ei8KvWlyQH", "en"),
    ("rafaela", "6k6cRt7QJfm5ChrH", "pt"),
    ("rebeca", "Sx8m6WdXBG06zwZT", "es"),
    ("regina", "s58clrg2fe0MO7Y-", "es"),
    ("russell", "_6Aslh2DxfmnRLmP", "en"),
    ("sir_bramwell", "y2eiKxtQo2B67deU", "en"),
    ("solene", "YhIHaAfQ0cQPDV9R", "fr"),
    ("spencer", "JOW0vchyswkQUMDS", "en"),
    ("svea", "Adoh4wA9tp5QXJi9", "de"),
    ("teddy", "Zn7g6NxLqXlJb7Qi", "en"),
    ("the_snow_queen", "NIWoEurJHMJjyUTe", "en"),
    ("the_thing_under_the_stairs", "GXaSqMEmOg12OIYb", "en"),
    ("tiphaine", "0fvH6XAKvA748v2v", "fr"),
    ("toby", "dME3IWyZBvmh1n1q", "en"),
    ("trevor", "0zLApsZfVLLNaEGl", "en"),
    ("ulises", "G5HLUsQd659UtRe7", "es"),
    ("vitor", "mWgyMRkFED70QxQv", "pt"),
    ("vyrmathrax_the_ancient", "xO8xQo3BrB8FuI6J", "en"),
    ("ximena", "VDwnGxAo68C8U8vC", "es"),
    ("yael", "ALczEazoF4n6PmGl", "es"),
    ("zoey", "NbpkqMVS3CJeq2j8", "en"),
)

_BY_SLUG: dict[str, tuple[str, str]] = {s: (i, lang) for s, i, lang in VOICE_CATALOG}


def slugify(name: str) -> str:
    """NFKD-strip diacritics, lowercase, each non-alphanumeric char -> ``_``."""
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]", "_", stripped)


# Catalog slug charset (full match). With the reserved ``DEFAULT_VOICE`` this is
# the whole slug rule; ``is_valid_slug`` is the single place it is applied.
SLUG_RE = re.compile(r"[a-z0-9_]+")


def is_valid_slug(slug: str) -> bool:
    """True for a non-empty ``[a-z0-9_]+`` slug that is not the reserved ``default``."""
    return SLUG_RE.fullmatch(slug) is not None and slug != DEFAULT_VOICE


def voice_catalog_drift(live_rows: list[dict]) -> dict[str, list[str]]:
    """Diff live ``voices.jsonl`` rows against the bundled catalog.

    Returns ``added`` / ``removed`` / ``changed`` slug lists, plus the live rows
    that break the catalog invariants (unique slugs passing ``is_valid_slug``):
    ``duplicate`` (a slug produced by more than one live row; the first row is the
    one diffed) and ``invalid`` (the raw names whose slug is empty or reserved).
    All five count as drift; none is collapsed.
    """
    live: dict[str, tuple[str, str]] = {}
    duplicate: list[str] = []
    invalid: list[str] = []
    for row in live_rows:
        name = str(row["name"])
        slug = slugify(name)
        if not is_valid_slug(slug):
            invalid.append(repr(name))
            continue
        if slug in live:
            if slug not in duplicate:
                duplicate.append(slug)
            continue
        live[slug] = (str(row["voice_id"]), str(row["language"]))
    return {
        "added": sorted(set(live) - set(_BY_SLUG)),
        "removed": sorted(set(_BY_SLUG) - set(live)),
        "changed": sorted(s for s in set(live) & set(_BY_SLUG) if live[s] != _BY_SLUG[s]),
        "duplicate": sorted(duplicate),
        "invalid": invalid,
    }


def voice_id(slug: str) -> str:
    return _BY_SLUG[slug][0]


def voice_language(slug: str) -> str:
    return _BY_SLUG[slug][1]


def voice_filename(slug: str) -> str:
    return f"voices/{voice_id(slug)}.safetensors"


def resolve_enabled(names) -> tuple[list[str], list[str]]:
    """Split ``names`` into ``(resolved, unknown)`` in input order, duplicates
    collapsed. ``resolved`` are catalog slugs; ``unknown`` are anything else
    (including the reserved ``default``, which is always advertised anyway)."""
    resolved: list[str] = []
    unknown: list[str] = []
    for name in names:
        if name in resolved or name in unknown:
            continue
        (resolved if name in _BY_SLUG else unknown).append(name)
    return resolved, unknown
