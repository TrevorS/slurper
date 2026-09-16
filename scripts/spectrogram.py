#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy", "soundfile", "matplotlib"]
# ///
"""Draws mel spectrograms of a folder of stems as one stacked figure for the README.

The mix goes on top, then vocals, drums, bass and other, all on the same dB scale relative to the
mix's peak, so a stem's level relative to the mix is visible.

Usage: uv run scripts/spectrogram.py <folder with mix.mp3, vocals.mp3, ...> [output.png]
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = ["mix", "vocals", "drums", "bass", "other"]
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


def main():
    folder = Path(sys.argv[1])
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else folder / "spectrogram.png"
    stems = {}
    for name in PANELS:
        audio, rate = sf.read(folder / f"{name}.mp3", dtype="float32")
        stems[name] = audio.mean(axis=1) if audio.ndim == 2 else audio
    filters, centers = mel_filters(rate)
    powers = {name: mel_power(audio, filters) for name, audio in stems.items()}
    peak = powers["mix"].max()
    seconds = len(stems["mix"]) / rate

    figure, axes = plt.subplots(len(PANELS), 1, figsize=(12, 2 * len(PANELS)), sharex=True, constrained_layout=True)
    ticks = [int(np.argmin(np.abs(centers - hz))) for hz in (100, 1_000, 10_000)]
    for axis, name in zip(axes, PANELS):
        db = 10 * np.log10(np.maximum(powers[name] / peak, 1e-12))
        axis.imshow(
            db, origin="lower", aspect="auto", cmap="magma", vmin=FLOOR_DB, vmax=0, extent=(0, seconds, 0, MELS)
        )
        axis.set_yticks(ticks, ["100 Hz", "1 kHz", "10 kHz"])
        axis.text(0.15, MELS - 6, name, color="white", fontsize=12, va="top")
    axes[-1].set_xlabel("seconds")
    figure.savefig(output, dpi=100)
    # An 8-bit palette is a third of the size and indistinguishable for a 256-step colormap.
    Image.open(output).convert("RGB").quantize(colors=256).save(output, optimize=True)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
