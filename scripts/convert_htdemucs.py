#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["torch==2.13.0", "coreai-torch==0.4.2", "demucs==4.1.0"]
# ///
"""Converts htdemucs to the Core AI model slurper runs (Sources/SlurperKit/Demucs.swift hosts it).

The graph is HTDemucs.forward from the normalization to just before _mask, for one 7.8 s segment:
mix[1,2,343980] + spec[1,4,2048,336] -> time[1,8,343980] + freq[1,16,2048,336]. `spec` is
`_magnitude(_spec(mix))`: left real, left imaginary, right real, right imaginary. The outputs are
denormalized; the host inverts `freq` with `_ispec` and adds `time`. The model stays float32: at float16
every output overflowed to NaN.

Before saving, the Core AI output on the GPU is checked against PyTorch on a synthetic segment.

Usage: uv run scripts/convert_htdemucs.py [output.aimodel]
"""

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

import coreai.runtime as rt
import numpy as np
import torch
from coreai_torch import TorchConverter, get_decomp_table
from demucs.pretrained import get_model

LENGTH = 343_980
SOURCES = 4
BINS, FRAMES = 2048, 336
DEFAULT_OUTPUT = Path.home() / "Library/Application Support/Slurper/Models/htdemucs-CoreAI/htdemucs_fp32.aimodel"
MINIMUM_SDR = 60


def std(x, dim):
    """torch.std (unbiased) without aten.var, which Core AI doesn't lower."""
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
    model = get_model("htdemucs").models[0].eval()
    mix = test_segment()
    with torch.no_grad():
        reference = model(mix)
        spec = model._magnitude(model._spec(mix))

    exported = torch.export.export(Core(model).eval(), args=(mix, spec)).run_decompositions(get_decomp_table())
    program = (
        TorchConverter()
        .add_exported_program(exported, input_names=["mix", "spec"], output_names=["time", "freq"])
        .to_coreai()
    )
    program.optimize()
    with tempfile.TemporaryDirectory() as workspace:
        staging = Path(workspace) / output.name
        program.save_asset(staging, rt.AIModelAssetMetadata())

        async def run():
            gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
            function = (await rt.AIModel.load(str(staging), gpu)).load_function("main")
            inputs = {
                "mix": rt.NDArray(np.ascontiguousarray(mix.numpy())),
                "spec": rt.NDArray(np.ascontiguousarray(spec.numpy())),
            }
            result = await function(inputs=inputs)
            return (torch.from_numpy(np.asarray(result[name].numpy()).astype(np.float32)) for name in ("time", "freq"))

        stems = finish(model, *asyncio.run(run()))
        scores = [sdr(reference[0, s], stems[0, s]) for s in range(SOURCES)]
        print(
            "Core AI vs PyTorch SDR:", ", ".join(f"{name} {score:.1f} dB" for name, score in zip(model.sources, scores))
        )
        if min(scores) < MINIMUM_SDR:
            sys.exit(f"Core AI output is below {MINIMUM_SDR} dB against PyTorch; not installing {output}")

        shutil.rmtree(output, ignore_errors=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staging, output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
