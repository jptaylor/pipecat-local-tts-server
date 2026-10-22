# mlx-audio: `gradium/phonon-31m-en` HEAD config uses an unsupported `model_type`

Draft only; not filed. The maintainer decides whether to open it.

## Problem

`gradium/phonon-31m-en` at HEAD (and the commit following the pinned one) sets
`model_type: gradium_phonon_tts` in `config.json`. `mlx_audio.tts.utils.load()` in
mlx-audio 0.6.1 (latest on PyPI at the time of writing) and on `main` raises:

    ValueError: Model type gradium_phonon_tts not supported for tts.

The Phonon model is supported by mlx-audio's `pocket_tts` module (PR #1049, first
released in 0.6.1), but only when `config.json` carries no `model_type` (the loader
then falls back to path/name routing to `pocket_tts`).

## Workaround in this repo

`tts_server/backends/phonon.py` pins the model to revision
`52433c46975bfa13144c56204c9f4ee1c5b148aa` (config without `model_type`) via
`snapshot_download(revision=...)` and loads the local path. The pin is repo-wide with
`mlx-audio==0.6.1`; it moves only in a deliberate bump together with an mlx-audio that
registers `gradium_phonon_tts` (or an alias to `pocket_tts`).

## Suggested upstream change

Register `gradium_phonon_tts` as an alias of the `pocket_tts` model type in the TTS
model-type resolver, so current Phonon checkpoints load without pinning an older
revision.
