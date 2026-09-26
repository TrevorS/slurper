#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["torch==2.13.0", "demucs==4.1.0", "musdb==0.4.2", "numpy"]
# ///
"""Scores slurper against PyTorch htdemucs and htdemucs_ft on the MUSDB18 7-second test previews.

SDR per stem is 10 log10(sum(s^2) / sum((s - estimate)^2)) over both channels of a track, averaged over the
tracks where every stem has audio (SDR is undefined for a silent reference). demucs runs without shifts and
with overlap 0.25, like slurper. musdb downloads the previews into <work dir>/musdb on the first run.

Usage: uv run scripts/benchmark.py <slurper binary> <work dir>
"""

import subprocess
import sys
from pathlib import Path

import musdb
import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

STEMS = ["vocals", "drums", "bass", "other"]


def write_wav(path, audio, rate):
    """A float32 WAV of [samples, channels] audio."""
    data = np.ascontiguousarray(audio, dtype="<f4").tobytes()
    channels = audio.shape[1]
    header = b"RIFF" + (36 + len(data)).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
    header += (3).to_bytes(2, "little") + channels.to_bytes(2, "little") + rate.to_bytes(4, "little")
    header += (
        (rate * channels * 4).to_bytes(4, "little") + (channels * 4).to_bytes(2, "little") + (32).to_bytes(2, "little")
    )
    path.write_bytes(header + b"data" + len(data).to_bytes(4, "little") + data)


def read_wav(path):
    """[samples, channels] from a float32 or int16 WAV."""
    data = Path(path).read_bytes()
    position, channels, bits, samples = 12, 0, 0, np.zeros((0, 2))
    while position + 8 <= len(data):
        chunk, size = data[position : position + 4], int.from_bytes(data[position + 4 : position + 8], "little")
        body = data[position + 8 : position + 8 + size]
        if chunk == b"fmt ":
            channels, bits = int.from_bytes(body[2:4], "little"), int.from_bytes(body[14:16], "little")
        elif chunk == b"data":
            samples = np.frombuffer(body, dtype=np.float32 if bits == 32 else np.int16).reshape(-1, channels)
        position += 8 + size + (size & 1)
    return samples.astype(np.float64)


def sdr(reference, estimate):
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]
    return 10 * np.log10((reference**2).sum() / max(((reference - estimate) ** 2).sum(), 1e-20))


def main():
    slurper, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    database = musdb.DB(root=str(work / "musdb"), download=True, subsets="test")
    tracks = [t for t in database if all((t.targets[s].audio ** 2).sum() > 1e-6 * len(t.audio) for s in STEMS)]
    seconds = sum(len(t.audio) / t.rate for t in tracks)
    print(f"{len(tracks)} of {len(database)} test tracks have every stem, {seconds:.0f} s", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    models = {name: get_model(name) for name in ["htdemucs", "htdemucs_ft"]}
    scores = {row: {s: [] for s in STEMS} for row in ["slurper", *models]}

    for index, track in enumerate(tracks):
        name = f"t{index:02d}"
        mix_path = work / "mixes" / f"{name}.wav"
        mix_path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(mix_path, track.audio, track.rate)

        subprocess.run([str(slurper), str(mix_path), "--out", str(work / "slurper")], check=True, capture_output=True)
        for s in STEMS:
            estimate = read_wav(work / "slurper" / name / f"{s}.wav")
            if s == "other":  # MUSDB18's other includes the horns slurper takes off first
                estimate = estimate + read_wav(work / "slurper" / name / "horns.wav")
            scores["slurper"][s].append(sdr(track.targets[s].audio, estimate))

        mix = torch.from_numpy(track.audio.T.astype(np.float32).copy())[None]
        for row, model in models.items():
            with torch.no_grad():
                out = apply_model(model, mix, shifts=0, split=True, overlap=0.25, progress=False, device=device)[0]
            for s in STEMS:
                scores[row][s].append(
                    sdr(track.targets[s].audio, out[model.sources.index(s)].numpy().T.astype(np.float64))
                )
        print(f"{name} {track.name}: " + ", ".join(f"{s} {scores['slurper'][s][-1]:.1f}" for s in STEMS), flush=True)

    print(f"\nSDR in dB, mean over {len(tracks)} tracks ({seconds:.0f} s):\n")
    print("| Pipeline | Vocals | Drums | Bass | Other | Average |")
    print("|---|---|---|---|---|---|")
    for row in scores:
        means = [float(np.mean(scores[row][s])) for s in STEMS]
        print(f"| {row} | " + " | ".join(f"{m:.2f}" for m in means) + f" | {np.mean(means):.2f} |")


if __name__ == "__main__":
    main()
