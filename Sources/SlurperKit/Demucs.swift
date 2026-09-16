import Accelerate
import CoreML
import Foundation

/// htdemucs (drums, bass, other, vocals; Meta, MIT) on Core ML.
///
/// The graph is the model's real-valued core for one 7.8 s segment: `mix[1,2,343980]` and its
/// spectrogram `spec[1,4,2048,336]` (left real, left imaginary, right real, right imaginary) in, the
/// denormalized time branch `time[1,8,343980]` and frequency branch `freq[1,16,2048,336]` out. This host
/// does what demucs does around it in PyTorch: `HTDemucs._spec`, `_ispec` of `freq` added to `time`, and
/// `apply_model`'s split into segments overlapping by a quarter with triangular crossfades (no shifts).
/// `scripts/convert_htdemucs.py` produced the model. Segments run one at a time.
final class Demucs: Sendable {
    static let sampleRate = 44_100
    static let sources = ["drums", "bass", "other", "vocals"]
    static let segment = 343_980
    private static let nFFT = 4096
    private static let hop = 1024
    private static let bins = nFFT / 2
    /// Spectrogram frames per segment, `ceil(segment / hop)`.
    private static let frames = (segment + hop - 1) / hop
    /// `_spec` reflect-pads by this much beyond `torch.stft`'s centering, so frames line up with the hop.
    private static let pad = hop / 2 * 3
    /// Periodic Hann.
    private static let window = (0..<nFFT).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(nFFT)) }
    /// One over the overlap-added squared window across a segment of `_ispec` output. torch.istft adds
    /// the squared windows of all 4 + `frames` frames, then drops `nFFT / 2` samples from the front.
    private static let inverseEnvelope: [Float] = {
        var envelope = [Float](repeating: 0, count: nFFT + hop * (frames + 3))
        for f in 0..<(frames + 4) {
            for k in 0..<nFFT { envelope[f * hop + k] += window[k] * window[k] }
        }
        let start = nFFT / 2 + pad
        return envelope[start..<start + segment].map { 1 / $0 }
    }()

    private let model: Model

    init(model url: URL) async throws {
        model = try await Model(package: url, inputs: ["mix", "spec"], outputs: ["time", "freq"])
    }

    /// Stems of stereo audio (`[left, right]` at 44.1 kHz), keyed by source name.
    func stems(of audio: [[Float]], progress: (Double) -> Void) async throws -> [String: [[Float]]] {
        let (length, segment) = (audio[0].count, Self.segment)
        let offsets = Array(stride(from: 0, to: length, by: segment * 3 / 4))
        let half = segment / 2
        let weight = (0..<segment).map { Float($0 < half ? $0 + 1 : segment - $0) / Float(half) }

        let silence = [[Float]](repeating: [Float](repeating: 0, count: length), count: 2)
        var sum = [[[Float]]](repeating: silence, count: Self.sources.count)
        var weights = [Float](repeating: 0, count: length)
        for (index, offset) in offsets.enumerated() {
            let stems = try await stems(ofSegment: Self.segment(of: audio, at: offset))
            let count = min(segment, length - offset)
            let trim = (segment - count) / 2
            for s in stems.indices {
                for c in 0..<2 {
                    for k in 0..<count { sum[s][c][offset + k] += weight[k] * stems[s][c][trim + k] }
                }
            }
            for k in 0..<count { weights[offset + k] += weight[k] }
            progress(Double(index + 1) / Double(offsets.count))
        }
        return Dictionary(uniqueKeysWithValues: Self.sources.indices.map { s in
            (Self.sources[s], sum[s].map { vDSP.divide($0, weights) })
        })
    }

    /// `TensorChunk.padded`: the `segment` samples centered on `[offset, offset + count)`, zero outside the audio.
    static func segment(of audio: [[Float]], at offset: Int) -> [[Float]] {
        let length = audio[0].count
        let start = offset - (segment - min(segment, length - offset)) / 2
        let (from, to) = (max(0, start), min(length, start + segment))
        return audio.map { channel in
            var x = [Float](repeating: 0, count: segment)
            x.replaceSubrange(from - start..<to - start, with: channel[from..<to])
            return x
        }
    }

    /// Stems of exactly one stereo segment: `[source][channel][sample]`.
    func stems(ofSegment mix: [[Float]]) async throws -> [[[Float]]] {
        let (segment, planes) = (Self.segment, Self.bins * Self.frames)
        var spec = [Float]()
        spec.reserveCapacity(4 * planes)
        for channel in mix {
            let (real, imaginary) = Self.spectrogram(channel)
            spec += real
            spec += imaginary
        }

        let outputs = try await model.run([
            "mix": MLTensor(shape: [1, 2, segment], scalars: mix[0] + mix[1]),
            "spec": MLTensor(shape: [1, 4, Self.bins, Self.frames], scalars: spec),
        ])
        guard let time = outputs["time"], let freq = outputs["freq"] else {
            throw SplitError.model("the Demucs model returned no time or freq output")
        }
        let timeValues = await time.values
        let freqValues = await freq.values

        return (0..<Self.sources.count).map { s in
            (0..<2).map { c in
                let base = (s * 4 + c * 2) * planes
                let wave = Self.inverseSpectrogram(
                    real: freqValues[base..<base + planes], imaginary: freqValues[base + planes..<base + 2 * planes])
                let start = (s * 2 + c) * segment
                return vDSP.add(wave, timeValues[start..<start + segment])
            }
        }
    }

    /// `HTDemucs._spec` of one segment: bins 0..<2048 by frames 2..<338 of a normalized, centered,
    /// reflect-padded STFT, as `[bin * frames + frame]` planes.
    static func spectrogram(_ x: [Float]) -> (real: [Float], imaginary: [Float]) {
        let (n, h) = (nFFT, hop)
        let padded = reflectPad(reflectPad(x, left: pad, right: pad + frames * h - x.count), left: n / 2, right: n / 2)
        let transform = try! vDSP.DiscreteFourierTransform(count: n, direction: .forward, transformType: .complexComplex, ofType: Float.self)
        let zeros = [Float](repeating: 0, count: n)
        var (frame, re, im) = (zeros, zeros, zeros)
        var real = [Float](repeating: 0, count: bins * frames), imaginary = real
        let scale = 1 / Float(n).squareRoot()
        for f in 0..<frames {
            let start = (f + 2) * h
            vDSP.multiply(padded[start..<start + n], window, result: &frame)
            transform.transform(inputReal: frame, inputImaginary: zeros, outputReal: &re, outputImaginary: &im)
            for bin in 0..<bins {
                real[bin * frames + f] = re[bin] * scale
                imaginary[bin * frames + f] = im[bin] * scale
            }
        }
        return (real, imaginary)
    }

    /// `HTDemucs._ispec` of one segment: the inverse of `spectrogram`, `segment` samples long.
    static func inverseSpectrogram(real: ArraySlice<Float>, imaginary: ArraySlice<Float>) -> [Float] {
        let (n, h) = (nFFT, hop)
        let transform = try! vDSP.DiscreteFourierTransform(count: n, direction: .inverse, transformType: .complexComplex, ofType: Float.self)
        let zeros = [Float](repeating: 0, count: n)
        var (inReal, inImaginary, outReal, outImaginary) = (zeros, zeros, zeros, zeros)
        // Output sample i is sample i + nFFT / 2 + pad of the overlap-add, where spectrogram frame f is frame f + 2.
        let origin = n / 2 + pad
        var output = [Float](repeating: 0, count: segment)
        let scale = 1 / Float(n).squareRoot()
        for f in 0..<frames {
            for bin in 0..<bins {
                inReal[bin] = real[real.startIndex + bin * frames + f]
                inImaginary[bin] = imaginary[imaginary.startIndex + bin * frames + f]
            }
            transform.transform(inputReal: inReal, inputImaginary: inImaginary, outputReal: &outReal, outputImaginary: &outImaginary)
            // irfft of a one-sided spectrum: twice the real part of the inverse, less the DC counted twice.
            let frameStart = (f + 2) * h - origin
            for k in max(0, -frameStart)..<min(n, segment - frameStart) {
                output[frameStart + k] += (2 * outReal[k] - inReal[0]) * scale * window[k]
            }
        }
        return vDSP.multiply(output, inverseEnvelope)
    }
}
