#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy", "soundfile", "matplotlib"]
# ///
"""Draws mel spectrograms of a folder of stems as one stacked figure for the README.

The mix goes on top, then whichever of vocals, horns, drums, bass and other the folder holds, all on the same
dB scale relative to the mix's peak, so a stem's level relative to the mix is visible.

With --transcription, the bar lines and chords from slurper's transcription/ folder are drawn over the mix,
shifted by --start, the excerpt's offset into the song in seconds.

Usage: uv run scripts/spectrogram.py <folder with mix.mp3, vocals.mp3, ...> [output.png]
           [--transcription DIR --start SECONDS]
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = ["mix", "vocals", "horns", "drums", "bass", "other"]
N_FFT, HOP, MELS = 2048, 512, 128
LOW, HIGH = 20.0, 16_000.0
FLOOR_DB = -80


def mel(frequency):
    return 2595 * np.log10(1 + frequency / 700)


def mel_filters(rate):
    """[MELS, N_FFT // 2 + 1] triangular filters spaced evenly on the mel scale."""
    edges = 700 * (10 ** (np.linspace(mel(LOW), mel(HIGH), MELS + 2) / 2595) - 1)
    bins = np.fft.rfftfreq(N_FFT, 1 / rate)
    filters = np.zeros((MELS, bins.size))
    for m in range(MELS):
        lower, center, upper = edges[m : m + 3]
        rising = (bins - lower) / (center - lower)
        falling = (upper - bins) / (upper - center)
        filters[m] = np.clip(np.minimum(rising, falling), 0, None)
    return filters, edges[1:-1]


def mel_power(audio, filters):
    """[MELS, frames] mel power of a mono signal."""
    window = np.hanning(N_FFT + 1)[:-1]
    frames = 1 + (len(audio) - N_FFT) // HOP
    starts = np.arange(frames) * HOP
    segments = audio[starts[:, None] + np.arange(N_FFT)] * window
    power = np.abs(np.fft.rfft(segments, axis=1)) ** 2
    return filters @ power.T


def read_lab(path, start, seconds):
    """Rows of a .lab file whose first column falls in the excerpt, with times relative to it."""
    rows = [line.split("\t") for line in path.read_text().splitlines() if line]
    return [(float(row[0]) - start, *row[1:]) for row in rows if start <= float(row[0]) < start + seconds]


def short_chord(label):
    """A lead-sheet name for a Harte chord label: Bb:maj is Bb, C:min7 is Cm7, F:7 is F7."""
    root, _, quality = label.partition(":")
    return root + {"maj": "", "min": "m", "min7": "m7"}.get(quality, quality)


def draw_transcription(axis, folder, start, seconds):
    """Dashed bar lines and chord names over the mix panel."""
    for time, *_ in read_lab(folder / "downbeat.lab", start, seconds):
        axis.axvline(time, color="white", linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.7)
    # A chord shorter than about a bar is left unlabeled, so neighbouring names don't overlap.
    previous, last_drawn = None, -1.0
    for time, _, label in read_lab(folder / "chord.lab", start, seconds):
        if label != previous and label != "N" and time - last_drawn >= 0.9:
            axis.text(
                time + 0.08,
                8,
                short_chord(label),
                color="white",
                fontsize=9,
                va="bottom",
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 1.5, "linewidth": 0},
            )
            last_drawn = time
        previous = label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--transcription", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    arguments = parser.parse_args()
    folder = arguments.folder
    output = arguments.output or folder / "spectrogram.png"
    panels = [name for name in PANELS if (folder / f"{name}.mp3").exists()]
    stems = {}
    for name in panels:
        audio, rate = sf.read(folder / f"{name}.mp3", dtype="float32")
        stems[name] = audio.mean(axis=1) if audio.ndim == 2 else audio
    filters, centers = mel_filters(rate)
    powers = {name: mel_power(audio, filters) for name, audio in stems.items()}
    peak = powers["mix"].max()
    seconds = len(stems["mix"]) / rate

    figure, axes = plt.subplots(len(panels), 1, figsize=(12, 2 * len(panels)), sharex=True, constrained_layout=True)
    ticks = [int(np.argmin(np.abs(centers - hz))) for hz in (100, 1_000, 10_000)]
    for axis, name in zip(axes, panels):
        db = 10 * np.log10(np.maximum(powers[name] / peak, 1e-12))
        axis.imshow(
            db, origin="lower", aspect="auto", cmap="magma", vmin=FLOOR_DB, vmax=0, extent=(0, seconds, 0, MELS)
        )
        axis.set_yticks(ticks, ["100 Hz", "1 kHz", "10 kHz"])
        axis.text(0.15, MELS - 6, name, color="white", fontsize=12, va="top")
    if arguments.transcription:
        draw_transcription(axes[0], arguments.transcription, arguments.start, seconds)
    axes[-1].set_xlabel("seconds")
    figure.savefig(output, dpi=100)
    # An 8-bit palette is a third of the size and indistinguishable for a 256-step colormap.
    Image.open(output).convert("RGB").quantize(colors=256).save(output, optimize=True)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
