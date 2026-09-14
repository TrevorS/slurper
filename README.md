# slurper

A command-line tool that takes a YouTube URL or a local audio file and writes the mix plus vocals, drums, bass, other and instrumental stems. It can also cut drum kits and bar-aligned loops from the stems, and write copies for the Elektron Digitakt II.

Version 0.1.0 (`slurper --version`).

## Requirements

- macOS 27 on Apple Silicon
- Xcode 27
- `brew install yt-dlp ffmpeg deno uv`

## Build, test, install

```
make build       # Release build in build/Build/Products/Release
make test        # unit tests with a coverage report
make install     # to ~/.local/bin (override with PREFIX=...)
make benchmark   # score against PyTorch demucs on the MUSDB18 test previews
make model       # rebuild the htdemucs Core AI model from PyTorch
```

`make model` runs `scripts/convert_htdemucs.py` through uv, which installs PyTorch, coreai-torch and demucs into its own environment; demucs downloads the htdemucs checkpoint. The converted model replaces the downloaded one in `~/Library/Application Support/Slurper/Models/htdemucs-CoreAI/`.

Pushing a `v*` tag builds `slurper-macos-arm64.zip`, which holds the binary, and attaches it to a GitHub release. The build is unsigned, so clear the quarantine flag after unzipping:

```
xattr -d com.apple.quarantine slurper
```

## Use

```
slurper "https://www.youtube.com/watch?v=..." --digitakt
slurper song.wav --out ~/Desktop/stems
slurper song.wav --kit drums --loops drums,bass --digitakt
```

Files are written as 44.1 kHz float WAVs to `~/Music/Slurper/Stems/<title>/`, where `/`, `:` and `\` in the title become `-`, and ` 2`, ` 3`, ... is appended when the folder exists. The first run downloads the 493 MB vocal model and the 169 MB htdemucs model into `~/Library/Application Support/Slurper/Models`.

Model loading runs alongside the download, vocal chunks run two at a time on the GPU, and each stem is written as soon as it exists. Stems named in `--loops` wait for the drums' bar lines.

## Stem quality

SDR in dB, mean over the 50 MUSDB18 test previews (340 s), M2 MacBook Air:

| Pipeline | Vocals | Drums | Bass | Other | Average |
|---|---|---|---|---|---|
| slurper | 11.28 | 9.89 | 8.62 | 6.18 | 8.99 |
| htdemucs_ft (PyTorch) | 8.54 | 9.56 | 8.89 | 4.98 | 7.99 |
| htdemucs (PyTorch) | 8.41 | 9.54 | 8.47 | 4.91 | 7.83 |

SDR here is 10 log10(Σs² / Σ(s − ŝ)²) over both channels of a track. demucs runs without shifts and with overlap 0.25, as slurper does. slurper's vocals come from Mel-Band RoFormer, and its drums, bass and other from htdemucs run on the mix minus those vocals. The previews are lossy 7-second excerpts, so absolute scores sit below published MUSDB18-HQ numbers. `make benchmark` reproduces the table.

## Digitakt II export

`Digitakt II/<prefix>_<stem>.wav`: 48 kHz, 16-bit dithered PCM, stereo (mono when both channels match), one file per stem. `<prefix>` is the title's ASCII letters and digits, words joined by underscores and cut to 20 characters, or `song` when none are left. Elektronauts users report imports failing above about 59 MB, roughly 5 minutes of stereo; Elektron does not document a limit.

## Kits and loops

`--kit` and `--loops` take any of `mix`, `vocals`, `instrumental`, `drums`, `bass` and `other`, comma-separated. Their files go next to the stem, and under `Digitakt II/` with the title prefix when `--digitakt` is on.

`--kit drums` writes `drums_kit/drums_01.wav`, `drums_02.wav`, ...: one example of each distinct hit, the most frequent first. Each hit is described by its levels in 24 mel bands in three windows ending 23, 46 and 93 ms after the attack, less its mean level so velocity doesn't split a sound. Hits within 6 dB RMS of a group's mean join it. Groups heard fewer than 3 times are dropped, unless no group reaches 3. Each group keeps, from the half of its hits nearest the group's mean, the one with the least tail from the hit before it and the most room before the next. A kit hit starts 3 ms before its attack, fades out over 5 ms, and runs at most 2 s.

`--loops drums,bass` writes `drums_loops_170bpm/drums_bar001.wav`, ...: 4-bar loops (`--bars N` to change), cut at the drum stem's bar lines so every stem's `bar017` lines up. Loops within 4.5 dB RMS of an earlier loop, compared by band levels at each 16th note, are left out, and so are silent ones. The tempo is the autocorrelation peak of the drums' onset envelope between 70 and 180 BPM, weighted toward 120 BPM, taking the faster of two tempos an octave apart when it correlates at least 80% as well. Beats come from Ellis's dynamic-programming tracker, the downbeat is the beat of the bar with the most onset strength below 150 Hz (assuming 4/4), and each bar line snaps to an attack within 30 ms. When the tempo lands on half or double what you count, pass `--bpm`. `--bars` and `--bpm` only apply with `--loops`.

Hits are found with spectral flux on the log magnitude (1024-sample frames, 256-sample hop, bins weighted per sixth octave so kicks aren't outvoted by cymbals), with an adaptive threshold, and placed to within one 64-sample block by the largest short-term level jump. Onsets closer than 50 ms merge, hits more than 40 dB below the stem's peak are dropped, and so are onsets where the level falls, which is what an abrupt stop looks like.

The thresholds come from synthetic drums and one 135 s drum and bass track, where the tracker found 170.00 BPM and every bar came within 17 ms of that tempo's bar length. On that track, kicks landing about 65 ms after another hit are missed (41 of 369 kick-band peaks). A 7 s loop at 144 BPM read as 72 BPM and needed `--bpm 144`. Nothing measures how steady a beat is, so drums without one can still produce a tempo and loops.

## Models

- Vocals: [Kim Mel-Band RoFormer](https://huggingface.co/KimberleyJSN/melbandroformer) (MIT weights), using the [Core AI conversion](https://huggingface.co/mlboydaisuke/MelBandRoformer-Vocal-CoreAI) from coreai-model-zoo.
- Drums, bass, other: [htdemucs](https://github.com/facebookresearch/demucs) (Meta, MIT) on Core AI, downloaded from [TrevorJS/htdemucs-CoreAI](https://huggingface.co/TrevorJS/htdemucs-CoreAI). `scripts/convert_htdemucs.py` exports the model's real-valued core for one 7.8 s segment with coreai-torch, in float32 because float16 overflows, and checks it against PyTorch before saving it. `Sources/SlurperKit/Demucs.swift` does the STFT, inverse STFT and segment crossfades around it, matching demucs's PyTorch code: on a 135 s song the stems match PyTorch's `apply_model` at 113-120 dB SDR. Segments run one at a time, because two at once on one Core AI model corrupt the outputs.
