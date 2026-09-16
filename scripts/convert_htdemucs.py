#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["torch==2.7.0", "coremltools==9.0", "demucs==4.1.0", "numpy<2"]
# ///
"""Converts htdemucs to the Core ML model slurper runs (Sources/SlurperKit/Demucs.swift hosts it).

The graph is HTDemucs.forward from the normalization to just before _mask, for one 7.8 s segment:
mix[1,2,343980] + spec[1,4,2048,336] -> time[1,8,343980] + freq[1,16,2048,336]. `spec` is
`_magnitude(_spec(mix))`: left real, left imaginary, right real, right imaginary. The outputs are
denormalized; the host inverts `freq` with `_ispec` and adds `time`. The model stays float32: at float16
every output overflowed to NaN.

Before saving, the Core ML output on the GPU is checked against PyTorch on a synthetic segment.

Usage: uv run scripts/convert_htdemucs.py [output.mlpackage]
"""

import shutil
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from demucs.pretrained import get_model

LENGTH = 343_980
SOURCES = 4
BINS, FRAMES = 2048, 336
DEFAULT_OUTPUT = Path.home() / "Library/Application Support/Slurper/Models/htdemucs-CoreML/htdemucs_fp32.mlpackage"
MINIMUM_SDR = 60


def std(x, dim):
    """torch.std (unbiased) written out, so the traced graph has no aten.var."""
    count = 1
    for d in dim:
        count *= x.shape[d]
    centered = x - x.mean(dim=dim, keepdim=True)
    return (centered.pow(2).sum(dim=dim, keepdim=True) / (count - 1)).sqrt()


class Core(torch.nn.Module):
    """HTDemucs.forward from the normalization to just before _mask, for one full-length segment."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, mix, spec):
        m = self.m
        x = spec
        B, _, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        deviation = std(x, (1, 2, 3))
        x = (x - mean) / (1e-5 + deviation)

        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = std(xt, (1, 2))
        xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)

        if m.crosstransformer:
            b, c, f, t = x.shape
            if m.bottom_channels:
                x = m.channel_upsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                x = m.channel_downsampler(x.reshape(b, -1, f * t)).reshape(b, -1, f, t)
                xt = m.channel_downsampler_t(xt)

        for idx, decode in enumerate(m.decoder):
            x, pre = decode(x, saved.pop(-1), lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    xt, _ = tdec(pre[:, :, 0], None, length_t)
                else:
                    xt, _ = tdec(xt, saved_t.pop(-1), length_t)

        x = x.view(B, SOURCES, -1, Fq, T) * deviation[:, None] + mean[:, None]
        xt = xt.view(B, SOURCES, -1, LENGTH) * stdt[:, None] + meant[:, None]
        return xt.reshape(B, SOURCES * 2, LENGTH), x.reshape(B, SOURCES * 4, Fq, T)


def finish(model, time_out, freq_out):
    """The host's half: invert the frequency branch and add the time branch -> [1, 4, 2, LENGTH]."""
    zout = model._mask(None, freq_out.float().view(1, SOURCES, 4, BINS, FRAMES))
    return time_out.float().view(1, SOURCES, 2, LENGTH) + model._ispec(zout, LENGTH)


def test_segment():
    """Decaying noise bursts and tones, so every stem has something in it."""
    generator = torch.Generator().manual_seed(0)
    t = torch.arange(LENGTH) / 44_100
    mix = 0.05 * torch.randn(2, LENGTH, generator=generator)
    for start in np.arange(0, 7.5, 0.5):
        envelope = torch.exp(-(t - start).clamp_min(0) * 20) * (t >= start)
        mix += 0.5 * envelope * torch.randn(2, LENGTH, generator=generator)
    mix += 0.2 * torch.sin(2 * torch.pi * 55 * t) + 0.1 * torch.sin(2 * torch.pi * 440 * t)
    return mix[None]


def sdr(reference, estimate):
    reference, estimate = reference.double(), estimate.double()
    return float(10 * torch.log10(reference.pow(2).sum() / (reference - estimate).pow(2).sum().clamp_min(1e-20)))


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    # The attention fast path is one fused op Core ML has no lowering for; the plain path is ordinary matmuls.
    torch.backends.mha.set_fastpath_enabled(False)
    model = get_model("htdemucs").models[0].eval()
    mix = test_segment()
    with torch.no_grad():
        reference = model(mix)
        spec = model._magnitude(model._spec(mix))
        traced = torch.jit.trace(Core(model).eval(), (mix, spec), check_trace=False)

    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="mix", shape=tuple(mix.shape), dtype=np.float32),
            ct.TensorType(name="spec", shape=tuple(spec.shape), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="time", dtype=np.float32), ct.TensorType(name="freq", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=ct.target.macOS15,
    )
    converted.short_description = "htdemucs (Meta, MIT) core for one 7.8 s segment; slurper does the STFT around it"
    converted.license = "MIT"

    with tempfile.TemporaryDirectory() as workspace:
        staging = Path(workspace) / output.name
        converted.save(str(staging))
        loaded = ct.models.MLModel(str(staging), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        result = loaded.predict({"mix": mix.numpy(), "spec": spec.numpy()})
        time_out, freq_out = (torch.from_numpy(np.asarray(result[name], dtype=np.float32)) for name in ("time", "freq"))
        stems = finish(model, time_out, freq_out)
        scores = [sdr(reference[0, s], stems[0, s]) for s in range(SOURCES)]
        print(
            "Core ML vs PyTorch SDR:", ", ".join(f"{name} {score:.1f} dB" for name, score in zip(model.sources, scores))
        )
        if min(scores) < MINIMUM_SDR:
            sys.exit(f"Core ML output is below {MINIMUM_SDR} dB against PyTorch; not installing {output}")

        shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(output.with_suffix(".mlmodelc"), ignore_errors=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staging, output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
