# Phonon smoke fixtures

- `story_long.txt` — one continuous first-person story, **every sentence unique** (a looped
  text can mask truncation or drift). Paragraphs are separated by blank lines. Used by
  `tests/smoke/phonon_phase0_gate.py` Q7/Q8 as nested sentence-boundary prefixes (~300, 500,
  1000, 2000, 3000, 4000 chars, then the full text). The opening paragraphs and the
  "almost missed my train" line are adapted from the author's own `full-spectrum-180s.txt`
  reference passage; the rest is original continuation. Keep sentences unique if you extend it
  (the gate asserts this).
