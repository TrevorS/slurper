#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#     "torch==2.7.0", "coremltools==9.0", "numpy<2", "einops", "beartype", "librosa",
#     "rotary-embedding-torch==0.3.5", "huggingface-hub",
# ]
# ///
"""Converts Kim Mel-Band RoFormer (vocals) to the Core ML model slurper runs (Sources/SlurperKit/VocalSeparator.swift).

The graph is `frames[1,2,801,2048] -> recon[1,2,801,2048]` for one 8 s chunk: windowed DFT as a constant
matmul, band split, the axial rotary transformer, mask estimator, band average as a constant matmul, the
complex mask multiply in real arithmetic, and the inverse DFT (window included) as another matmul. The host
reflect-pads and frames the chunk, then overlap-adds `recon` and divides by the summed squared window. The
recipe follows coreai-model-zoo's melband_roformer export (BSD-3-Clause, Daisuke Majima).

Before saving, the Core ML output on the GPU is checked against the PyTorch model on a real 8 s chunk, and
that chunk and PyTorch's vocals are written next to the model as golden_raw.f32 and golden_vocals.f32
(stereo, channel-major float32) for the Swift tests.

Usage: uv run scripts/convert_melband_roformer.py [output.mlpackage]
"""

import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

SAMPLE_RATE = 44_100
CHUNK = 352_800
N_FFT, HOP, PAD = 2048, 441, 1024
BINS = N_FFT // 2 + 1
FRAMES = 1 + (CHUNK + 2 * PAD - N_FFT) // HOP  # 801
DEFAULT_OUTPUT = (
    Path.home() / "Library/Application Support/Slurper/Models/MelBandRoformer-Vocal-CoreML/mbr_fp16.mlpackage"
)
MINIMUM_COSINE = 0.9999

WEIGHTS = ("KimberleyJSN/melbandroformer", "MelBandRoformer.ckpt")
# The reference implementation the checkpoint was trained with.
KIM_CODE = "https://raw.githubusercontent.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/25f44ffb55ee3c301281bba21b2d6d311cb69ae2"
KIM_CONFIG = {
    "dim": 384,
    "depth": 6,
    "stereo": True,
    "num_stems": 1,
    "time_transformer_depth": 1,
    "freq_transformer_depth": 1,
    "num_bands": 60,
    "dim_head": 64,
    "heads": 8,
    "attn_dropout": 0,
    "ff_dropout": 0,
    "flash_attn": True,
    "dim_freqs_in": 1025,
    "sample_rate": SAMPLE_RATE,
    "stft_n_fft": N_FFT,
    "stft_hop_length": HOP,
    "stft_win_length": N_FFT,
    "stft_normalized": False,
    "mask_estimator_depth": 2,
}
# An 8 s stereo excerpt with vocals, published with the Core AI conversion of this model.
GOLDEN = ("mlboydaisuke/MelBandRoformer-Vocal-CoreAI", "5cf0e04d08569f4d2f5d89d500a8fb943894330f")


def load_reference(workspace):
    """The Kim repository's MelBandRoformer with its checkpoint, attention routed through F.scaled_dot_product_attention."""
    package = workspace / "kim/models/mel_band_roformer"
    package.mkdir(parents=True)
    (workspace / "kim/models/__init__.py").touch()
    for name in ("__init__.py", "attend.py", "mel_band_roformer.py"):
        urllib.request.urlretrieve(f"{KIM_CODE}/models/mel_band_roformer/{name}", package / name)
    sys.path.insert(0, str(workspace / "kim"))
    from models.mel_band_roformer import attend
    from models.mel_band_roformer.mel_band_roformer import MelBandRoformer

    attend.Attend.flash_attn = lambda self, q, k, v: F.scaled_dot_product_attention(q, k, v)
    model = MelBandRoformer(**KIM_CONFIG).eval()
    state = torch.load(hf_hub_download(*WEIGHTS), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        sys.exit(f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model


class FixedRotary(nn.Module):
    """rotary_embedding_torch's rotation for one sequence length, with its angles as constants."""

    def __init__(self, rotary, length):
        super().__init__()
        with torch.no_grad():
            angles = rotary.forward(torch.arange(length)).float()  # [length, dim_head], each angle twice
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def rotate_queries_or_keys(self, t):
        pairs = t.reshape(*t.shape[:-1], -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).reshape(t.shape)
        return t * self.cos + rotated * self.sin


class Core(nn.Module):
    """MelBandRoformer.forward between the STFT and the iSTFT, in real arithmetic on the reference's submodules."""

    def __init__(self, model):
        super().__init__()
        self.band_split = model.band_split
        self.layers = model.layers
        self.mask_estimator = model.mask_estimators[0]
        indices = model.freq_indices.long()
        self.register_buffer("freq_indices", indices, persistent=False)
        # Band average: out[f] = sum of the masks at f / number of bands covering f.
        f2 = model.num_bands_per_freq.numel() * model.audio_channels
        average = torch.zeros(f2, indices.numel())
        average[indices, torch.arange(indices.numel())] = 1
        average /= model.num_bands_per_freq.repeat_interleave(model.audio_channels).clamp(min=1e-8)[:, None]
        self.register_buffer("average_t", average.t().contiguous(), persistent=False)  # [selected, f2]

        for time_transformer, freq_transformer in self.layers:
            for transformer, length in (
                (time_transformer, FRAMES),
                (freq_transformer, len(model.band_split.dim_inputs)),
            ):
                for attention, _ in transformer.layers:
                    attention.rotary_embed = FixedRotary(attention.rotary_embed, length)

    def forward(self, spec):
        # spec: [b, f2, t, 2] with f2 = (bin, channel) interleaved, last axis real/imaginary.
        b = spec.shape[0]
        x = spec.index_select(1, self.freq_indices)  # [b, selected, t, 2]
        x = x.permute(0, 2, 1, 3).reshape(b, FRAMES, -1)  # [b, t, selected * 2]
        x = self.band_split(x)  # [b, t, bands, dim]
        bands = x.shape[2]
        for time_transformer, freq_transformer in self.layers:
            x = x.permute(0, 2, 1, 3).reshape(b * bands, FRAMES, -1)
            x = time_transformer(x)
            x = x.reshape(b, bands, FRAMES, -1).permute(0, 2, 1, 3).reshape(b * FRAMES, bands, -1)
            x = freq_transformer(x)
            x = x.reshape(b, FRAMES, bands, -1)
        mask = self.mask_estimator(x)  # [b, t, selected * 2]
        mask = mask.reshape(b, FRAMES, -1, 2).permute(0, 3, 1, 2)  # [b, 2, t, selected]
        mask = torch.matmul(mask, self.average_t).permute(0, 3, 2, 1)  # [b, f2, t, 2]
        real = spec[..., 0] * mask[..., 0] - spec[..., 1] * mask[..., 1]
        imaginary = spec[..., 0] * mask[..., 1] + spec[..., 1] * mask[..., 0]
        return torch.stack((real, imaginary), dim=-1)


class Full(nn.Module):
    """frames[b,2,801,2048] -> recon[b,2,801,2048]: windowed DFT, Core, inverse DFT with the synthesis window."""

    def __init__(self, model):
        super().__init__()
        self.core = Core(model)
        n = np.arange(N_FFT)
        window = 0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)  # periodic Hann, torch.hann_window's default
        angle = 2 * np.pi * np.outer(n, np.arange(BINS)) / N_FFT  # [N_FFT, BINS]
        self.register_buffer(
            "dft_real", torch.tensor(window[:, None] * np.cos(angle), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "dft_imaginary", torch.tensor(-window[:, None] * np.sin(angle), dtype=torch.float32), persistent=False
        )
        weight = np.full(BINS, 2.0)
        weight[0] = weight[-1] = 1  # DC and Nyquist appear once in a one-sided spectrum
        synthesis = window[None, :] * weight[:, None] / N_FFT  # [BINS, N_FFT]
        self.register_buffer(
            "idft_real", torch.tensor(synthesis * np.cos(angle.T), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "idft_imaginary", torch.tensor(-synthesis * np.sin(angle.T), dtype=torch.float32), persistent=False
        )

    def forward(self, frames):
        b = frames.shape[0]
        real = torch.matmul(frames, self.dft_real)  # [b, 2, t, bins]
        imaginary = torch.matmul(frames, self.dft_imaginary)
        spec = torch.stack((real, imaginary), dim=-1).permute(0, 3, 1, 2, 4).reshape(b, BINS * 2, FRAMES, 2)
        masked = self.core(spec).reshape(b, BINS, 2, FRAMES, 2).permute(0, 2, 3, 1, 4)  # [b, 2, t, bins, 2]
        return torch.matmul(masked[..., 0], self.idft_real) + torch.matmul(masked[..., 1], self.idft_imaginary)


def frame(audio):
    """[2, CHUNK] -> [1, 2, FRAMES, N_FFT], the host's reflect padding and framing."""
    padded = np.pad(audio, ((0, 0), (PAD, PAD)), mode="reflect")
    starts = np.arange(FRAMES) * HOP
    return np.stack([padded[c][starts[:, None] + np.arange(N_FFT)] for c in range(2)])[None].astype(np.float32)


def overlap_add(recon):
    """[2, FRAMES, N_FFT] -> [2, CHUNK], the host's overlap-add, window normalization and trim."""
    total = N_FFT + HOP * (FRAMES - 1)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    out = np.zeros((2, total))
    weight = np.zeros(total)
    for i in range(FRAMES):
        out[:, i * HOP : i * HOP + N_FFT] += recon[:, i]
        weight[i * HOP : i * HOP + N_FFT] += window**2
    return out / np.maximum(weight, 1e-8)[None]


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    with np.errstate(all="ignore"):  # numpy 1.26 on Accelerate raises spurious FP flags in dot products
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def sdr(reference, estimate):
    reference, estimate = np.asarray(reference, dtype=np.float64), np.asarray(estimate, dtype=np.float64)
    return float(10 * np.log10((reference**2).sum() / max(((reference - estimate) ** 2).sum(), 1e-20)))


def golden_raw():
    path = hf_hub_download(GOLDEN[0], "golden_raw.f32", revision=GOLDEN[1])
    return np.fromfile(path, dtype=np.float32).reshape(2, CHUNK)


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    with tempfile.TemporaryDirectory() as workspace:
        workspace = Path(workspace)
        model = load_reference(workspace)
        raw = golden_raw()
        with torch.no_grad():
            reference = model(torch.from_numpy(raw)[None])[0].numpy()  # [2, CHUNK]
            full = Full(model).eval()
            frames = frame(raw)
            torch_vocals = overlap_add(full(torch.from_numpy(frames))[0].numpy())[:, PAD : PAD + CHUNK]
            print(f"PyTorch frames->recon vs MelBandRoformer.forward: cosine {cosine(torch_vocals, reference):.7f}")
            traced = torch.jit.trace(full, torch.from_numpy(frames), check_trace=False)

        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name="frames", shape=frames.shape, dtype=np.float32)],
            outputs=[ct.TensorType(name="recon", dtype=np.float32)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
            minimum_deployment_target=ct.target.macOS15,
        )
        converted.short_description = "Kim Mel-Band RoFormer vocals (MIT) for one 8 s chunk of STFT frames"
        converted.license = "MIT"

        staging = workspace / output.name
        converted.save(str(staging))
        loaded = ct.models.MLModel(str(staging), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        recon = np.asarray(loaded.predict({"frames": frames})["recon"], dtype=np.float32)[0]
        vocals = overlap_add(recon)[:, PAD : PAD + CHUNK]
        score = cosine(vocals, reference)
        print(f"Core ML fp16 vs PyTorch: cosine {score:.7f}, SDR {sdr(reference, vocals):.1f} dB")
        if score < MINIMUM_COSINE:
            sys.exit(f"Core ML output is below cosine {MINIMUM_COSINE} against PyTorch; not installing {output}")

        shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(output.with_suffix(".mlmodelc"), ignore_errors=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staging, output)
        raw.astype(np.float32).tofile(output.parent / "golden_raw.f32")
        reference.astype(np.float32).tofile(output.parent / "golden_vocals.f32")
    print(f"saved {output} and golden_raw.f32, golden_vocals.f32")


if __name__ == "__main__":
    main()
