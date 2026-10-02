#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "torch==2.7.0", "torchaudio==2.7.0", "coremltools==9.0", "numpy<2", "transformers==4.45.2",
#     "huggingface-hub==0.36.0", "safetensors", "scipy", "mir_eval==0.8.2",
# ]
# ///
"""Converts SheetSage2 (m-a-p, CC BY-NC 4.0 weights) to the two Core ML models slurper's --transcribe runs
(Sources/SlurperKit/SheetSage.swift hosts them).

The encoder is `samples[1,7200000] -> cross[12,8,7500,64]` for one 300 s window of 24 kHz mono audio, zero
padded: torchaudio's log-mel frontend as a strided convolution with constant DFT weights, MERT-v2-FullSong with
SheetSage2's attention adapters merged, the softmax-weighted mix of its 25 hidden states, the projection to the
decoder's width, and each decoder layer's cross-attention keys and values (layer i's keys at 2i, values at
2i + 1). It stays float32: at float16 the decoded tokens changed 27 tokens into a test song, and so did int8
weights; float32 matched PyTorch at 94.5 dB.

The decoder is one step of the 6-layer BART decoder, `token[1,1] + position[1] + mask[1,1,1,L] -> logits[1,V]`,
in float16 with its caches as Core ML states: self_k_i/self_v_i [1,8,5120,64], which each step writes at
`position`, and cross_k_i/cross_v_i [1,8,7500,64], which the host copies from the encoder once per window.
`mask` is zeros of length position + 1. Its length places the cache write: a slice whose bounds come from a
traced size converts to a state update, while indexing by the position tensor (index_put) converts to a
slice_update whose result Core ML never stores. The host masks the logits with the event grammar and takes
the argmax.

Before installing, the script transcribes 30 s of the README's "Swansong" excerpt with PyTorch and with the
Core ML models, and requires identical tokens. That audio (decoded the way SheetSage2 decodes files) and the
tokens are written next to the models as golden_audio.f32 and golden_tokens.json for the Swift tests, along
with the model card.

Usage: uv run scripts/convert_sheetsage2.py [output folder]
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn
from torch.nn import functional as F

REPO, REVISION = "m-a-p/SheetSage2", "398b22834dac7dd05e09b9c4e40a39fc479ec502"
SAMPLE_RATE = 24_000
WINDOW = 300 * SAMPLE_RATE
N_FFT, HOP, PAD = 2048, 240, 1024
BINS = N_FFT // 2 + 1
FRAMES = 7500
MAX_TOKENS = 5120
HEADS, HEAD = 8, 64
GOLDEN_SOURCE = Path(__file__).resolve().parent.parent / "assets/swansong/mix.mp3"
DEFAULT_OUTPUT = Path.home() / "Library/Application Support/Slurper/Models/SheetSage2-CoreML"
MINIMUM_SNR = 60


def load_model():
    """SheetSage2 with its adapters merged into the pinned MERT-v2 parent, imported from the snapshot as a package
    (transformers' remote-code loader leaves out modules that are only imported indirectly)."""
    snapshot = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=["*.py", "*.json", "model.safetensors"]))
    spec = importlib.util.spec_from_file_location(
        "sheetsage2", snapshot / "__init__.py", submodule_search_locations=[str(snapshot)]
    )
    sys.modules["sheetsage2"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules["sheetsage2"])
    from sheetsage2.modeling_sheetsage2 import SheetSage2Model

    return SheetSage2Model.from_pretrained(str(snapshot)).eval()


class Encoder(nn.Module):
    """samples[1, WINDOW] -> cross[12, HEADS, FRAMES, HEAD]: get_audio_features plus the cross-attention projections."""

    def __init__(self, model):
        super().__init__()
        frontend = model.encoder.feature_extractor
        n = np.arange(N_FFT)
        window = 0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)  # periodic Hann, torchaudio's default
        angle = 2 * np.pi * np.outer(np.arange(BINS), n) / N_FFT  # [BINS, N_FFT]
        dft = np.concatenate([window * np.cos(angle), -window * np.sin(angle)])[:, None, :]
        self.register_buffer("dft", torch.tensor(dft, dtype=torch.float32), persistent=False)
        self.register_buffer("filterbank", frontend.mel_scale.fb.clone(), persistent=False)  # [BINS, 128]
        self.register_buffer("mel_mean", frontend.mel_mean.clone(), persistent=False)
        self.register_buffer("mel_std", frontend.mel_std.clamp_min(1e-5), persistent=False)

        self.subsampling = model.encoder.subsampling_module
        self.layers = model.encoder.layers
        self.projection = model.encoder_projection
        self.register_buffer("weights", torch.softmax(model.layer_weight.detach(), 0), persistent=False)
        cos, sin = model.encoder.embed_positions(torch.zeros(1, FRAMES, 1))
        self.register_buffer("cos", cos.clone(), persistent=False)
        self.register_buffer("sin", sin.clone(), persistent=False)
        self.cross = nn.ModuleList(
            nn.ModuleList([layer.encoder_attn.k_proj, layer.encoder_attn.v_proj]) for layer in model.decoder.layers
        )

    def forward(self, samples):
        # torchaudio's Spectrogram (centered, reflect padded, power 2), MelScale and AmplitudeToDB (amin 1e-10).
        spectrum = F.conv1d(F.pad(samples[:, None], (PAD, PAD), mode="reflect"), self.dft, stride=HOP)
        power = spectrum[:, :BINS] ** 2 + spectrum[:, BINS:] ** 2
        mel = 10 * torch.log10(torch.matmul(power.transpose(1, 2), self.filterbank).clamp_min(1e-10))
        mel = (mel[:, :-1] - self.mel_mean) / self.mel_std

        hidden = self.subsampling(mel)
        mixed = hidden * self.weights[0]
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, (self.cos, self.sin))
            mixed = mixed + hidden * self.weights[i + 1]
        memory = self.projection(mixed)
        return torch.cat(
            [
                projection(memory).view(FRAMES, HEADS, HEAD).transpose(0, 1)[None]
                for pair in self.cross
                for projection in pair
            ]
        )


class DecoderStep(nn.Module):
    """One token through BartDecoder, with the self- and cross-attention caches as buffers (Core ML states)."""

    def __init__(self, model):
        super().__init__()
        self.embed = model.token_embedding
        self.positions = model.decoder.embed_positions  # BartLearnedPositionalEmbedding: offset 2
        self.norm = model.decoder.layernorm_embedding
        self.layers = model.decoder.layers
        for i in range(len(self.layers)):
            for name, length in (
                ("self_k", MAX_TOKENS),
                ("self_v", MAX_TOKENS),
                ("cross_k", FRAMES),
                ("cross_v", FRAMES),
            ):
                self.register_buffer(f"{name}_{i}", torch.zeros(1, HEADS, length, HEAD))

    @staticmethod
    def heads(x):
        return x.view(1, -1, HEADS, HEAD).transpose(1, 2)

    def forward(self, token, position, mask):
        x = self.embed(token) + F.embedding(position.long() + 2, self.positions.weight)[None]
        x = self.norm(x)
        length = mask.shape[-1]
        for i, layer in enumerate(self.layers):
            attention = layer.self_attn
            self_k, self_v = getattr(self, f"self_k_{i}"), getattr(self, f"self_v_{i}")
            self_k[:, :, length - 1 : length] = self.heads(attention.k_proj(x))
            self_v[:, :, length - 1 : length] = self.heads(attention.v_proj(x))
            out = F.scaled_dot_product_attention(
                self.heads(attention.q_proj(x)), self_k[:, :, :length], self_v[:, :, :length], attn_mask=mask
            )
            x = layer.self_attn_layer_norm(x + attention.out_proj(out.transpose(1, 2).reshape(1, 1, -1)))
            attention = layer.encoder_attn
            out = F.scaled_dot_product_attention(
                self.heads(attention.q_proj(x)), getattr(self, f"cross_k_{i}"), getattr(self, f"cross_v_{i}")
            )
            x = layer.encoder_attn_layer_norm(x + attention.out_proj(out.transpose(1, 2).reshape(1, 1, -1)))
            x = layer.final_layer_norm(x + layer.fc2(F.gelu(layer.fc1(x))))
        return x[0] @ self.embed.weight.T


def convert_encoder(model):
    samples = torch.zeros(1, WINDOW)
    with torch.no_grad():
        traced = torch.jit.trace(Encoder(model).eval(), (samples,), check_trace=False)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="samples", shape=tuple(samples.shape), dtype=np.float32)],
        outputs=[ct.TensorType(name="cross", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=ct.target.macOS15,
    )
    converted.short_description = "SheetSage2 encoder (MERT-v2-FullSong + adapters) for one 300 s window at 24 kHz"
    converted.license = "CC BY-NC 4.0"
    return converted


def convert_decoder(model):
    step = DecoderStep(model).eval()
    example = (torch.tensor([[1]], dtype=torch.int32), torch.tensor([1], dtype=torch.int32), torch.zeros(1, 1, 1, 2))
    with torch.no_grad():
        traced = torch.jit.trace(step, example, check_trace=False)
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="token", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="position", shape=(1,), dtype=np.int32),
            ct.TensorType(name="mask", shape=(1, 1, 1, ct.RangeDim(1, MAX_TOKENS, default=1)), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        states=[
            ct.StateType(wrapped_type=ct.TensorType(shape=tuple(buffer.shape), dtype=np.float16), name=name)
            for name, buffer in step.named_buffers()
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=ct.target.macOS15,
    )
    converted.short_description = "SheetSage2 decoder step with its caches as states"
    converted.license = "CC BY-NC 4.0"
    return converted


def golden_audio():
    """30 s of the Swansong excerpt as SheetSage2 reads files: ffmpeg to 24 kHz mono float32."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(GOLDEN_SOURCE),
        "-vn",
        "-t",
        "30",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    return np.frombuffer(subprocess.run(command, capture_output=True, check=True).stdout, dtype="<f4").copy()


def reference_tokens(model, audio):
    """PyTorch's tokens for one window, as SheetSage2's pipeline generates the only window of a short song."""
    from sheetsage2.generation_sheetsage2 import FULL_TASK_PROMPTS, constrained_prompt_generate

    segment = torch.from_numpy(np.pad(audio, (0, WINDOW - len(audio))))[None]
    tokens = constrained_prompt_generate(
        model,
        segment,
        FULL_TASK_PROMPTS,
        model.max_output_seq_len,
        autocast_dtype=None,
        stop_time_seconds=len(audio) / SAMPLE_RATE,
    )
    with torch.no_grad():
        memory = model.get_audio_features(segment).last_hidden_state
        cross = torch.cat(
            [
                projection(memory).view(FRAMES, HEADS, HEAD).transpose(0, 1)[None]
                for layer in model.decoder.layers
                for projection in (layer.encoder_attn.k_proj, layer.encoder_attn.v_proj)
            ]
        )
    return tokens.tolist(), cross.numpy()


def coreml_tokens(model, encoder, decoder, audio):
    """The Swift host's loop: encode, copy the cross keys and values into the decoder's state, then one step per
    token, masking each step's logits with the event grammar."""
    from sheetsage2.generation_sheetsage2 import FULL_TASK_PROMPTS, PromptGrammarState

    tokenizer = model.tokenizer
    samples = np.pad(audio, (0, WINDOW - len(audio)))[None]
    cross = np.asarray(encoder.predict({"samples": samples})["cross"], dtype=np.float32)
    state = decoder.make_state()
    for i in range(len(cross) // 2):
        state.write_state(f"cross_k_{i}", np.ascontiguousarray(cross[2 * i][None]))
        state.write_state(f"cross_v_{i}", np.ascontiguousarray(cross[2 * i + 1][None]))

    prefix = tokenizer.prompt_prefix(tokenizer.normalize_prompts(FULL_TASK_PROMPTS))
    grammar = PromptGrammarState(tokenizer)
    stop_time = len(audio) / SAMPLE_RATE
    tokens, mask = list(prefix), np.zeros((1, 1, 1, MAX_TOKENS), dtype=np.float16)
    for position in range(MAX_TOKENS):
        logits = decoder.predict(
            {
                "token": np.array([[tokens[position]]], dtype=np.int32),
                "position": np.array([position], dtype=np.int32),
                "mask": mask[..., : position + 1],
            },
            state=state,
        )["logits"]
        if position + 1 < len(tokens):
            continue
        allowed = grammar.allowed("cpu").numpy()
        token = int(np.where(allowed, np.asarray(logits).reshape(-1), -np.inf).argmax())
        tokens.append(token)
        finished = grammar.update(token)
        if (
            not finished
            and tokenizer.time_token_start <= token < tokenizer.time_token_end
            and tokenizer.token_to_time_id(token) / tokenizer.time_hz >= stop_time
        ):
            tokens.append(tokenizer.eos_token)
            finished = True
        if finished:
            break
    else:
        tokens.append(tokenizer.eos_token)
    return tokens, cross


MODEL_CARD = """---
license: cc-by-nc-4.0
base_model: m-a-p/SheetSage2
tags: [coreml, music-transcription]
---

# SheetSage2 on Core ML

[SheetSage2](https://huggingface.co/m-a-p/SheetSage2) (revision `{revision}`) and its parent encoder
[MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong), converted to Core ML for
[slurper](https://github.com/TrevorS/slurper) by `scripts/convert_sheetsage2.py`. The weights keep their
CC BY-NC 4.0 license: non-commercial use only, with attribution to the SheetSage2 and MERT-v2 authors.

- `encoder_fp32.mlpackage`: `samples[1,7200000]` (300 s of 24 kHz mono, zero padded) `-> cross[12,8,7500,64]`,
  the log-mel frontend, MERT-v2 with SheetSage2's adapters merged, the layer mix and projection, and each
  decoder layer's cross-attention keys (2i) and values (2i + 1). float32.
- `decoder_fp16.mlpackage`: one step of the 6-layer BART decoder, `token[1,1] + position[1] + mask[1,1,1,L]
  -> logits[1,31678]`, with self- and cross-attention caches as states. `mask` is zeros of length position + 1.
- `golden_audio.f32`, `golden_tokens.json`: 30 s of "Swansong" by Josh Woodward (CC BY 4.0) at 24 kHz mono, and
  the tokens PyTorch decodes from it, which these models reproduce exactly.

Please cite the SheetSage2 technical report (Jiang et al., 2026) and MERT (Li et al., ICLR 2024).
"""


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    model = load_model()
    audio = golden_audio()
    reference, reference_cross = reference_tokens(model, audio)
    print(f"PyTorch: {len(reference)} tokens")

    with tempfile.TemporaryDirectory() as workspace:
        staging = Path(workspace) / output.name
        staging.mkdir()
        convert_encoder(model).save(str(staging / "encoder_fp32.mlpackage"))
        convert_decoder(model).save(str(staging / "decoder_fp16.mlpackage"))
        encoder = ct.models.MLModel(str(staging / "encoder_fp32.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        decoder = ct.models.MLModel(str(staging / "decoder_fp16.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        tokens, cross = coreml_tokens(model, encoder, decoder, audio)

        reference_cross = reference_cross.astype(np.float64)
        snr = 10 * np.log10((reference_cross**2).sum() / ((reference_cross - cross) ** 2).sum())
        same = next((i for i, (a, b) in enumerate(zip(tokens, reference)) if a != b), min(len(tokens), len(reference)))
        print(f"Core ML vs PyTorch: cross keys and values {snr:.1f} dB, {len(tokens)} tokens, first {same} identical")
        if snr < MINIMUM_SNR or tokens != reference:
            sys.exit(f"Core ML does not reproduce PyTorch; not installing {output}")

        audio.astype("<f4").tofile(staging / "golden_audio.f32")
        (staging / "golden_tokens.json").write_text(json.dumps(reference))
        (staging / "README.md").write_text(MODEL_CARD.format(revision=REVISION))
        shutil.rmtree(output, ignore_errors=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staging, output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
