import CoreML
import Foundation

/// A RoFormer stem separator on Core ML: Kim Mel-Band RoFormer (vocals, MIT weights) and the MVSep Mega
/// 53-stem BS-RoFormer's wind stem (horns).
///
/// Each model folds STFT, the RoFormer and iSTFT into one fixed-shape graph,
/// `frames[1,2,F,2048] -> recon[1,2,F,2048]`, for one 8 s chunk framed at the model's hop: 801 frames at
/// hop 441 (vocals), 690 at hop 512 (horns). This host reflect-pads and frames each chunk, overlap-adds the
/// reconstruction divided by the summed squared window, and crossfades chunks across the song. The convert
/// scripts produced the models; the procedure follows the coreai-model-zoo reference host (BSD-3-Clause,
/// Daisuke Majima).
final class RoformerSeparator: Sendable {
    static let sampleRate = 44_100
    static let chunkSamples = 352_800
    private static let nFFT = 2048
    private static let pad = 1024

    private let model: Model
    private let hop: Int
    private let frames: Int
    private let windowSquareSum: [Float]
    private let chunksAtOnce: Int

    convenience init(_ model: StemSplitter.RoformerModel, chunksAtOnce: Int) async throws {
        try await self.init(model: model.package, hop: model.hop, chunksAtOnce: chunksAtOnce)
    }

    init(model url: URL, hop: Int, chunksAtOnce: Int) async throws {
        model = try await Model(package: url, inputs: ["frames"], outputs: ["recon"])
        self.hop = hop
        self.chunksAtOnce = max(chunksAtOnce, 1)

        let n = Self.nFFT
        let f = 1 + (Self.chunkSamples + 2 * Self.pad - n) / hop
        frames = f
        // The graph is fixed-shape, so a hop that disagrees with the model would only fail inside Core ML,
        // after the whole previous stage has run.
        if let shape = model.shape(ofInput: "frames"), shape != [1, 2, f, n] {
            throw SplitError.model("\(url.lastPathComponent) takes frames \(shape), not [1, 2, \(f), \(n)] at hop \(hop)")
        }
        let window = (0..<n).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(n)) }  // periodic Hann
        var sum = [Float](repeating: 0, count: n + hop * (f - 1))
        for i in 0..<f {
            for k in 0..<n { sum[i * hop + k] += window[k] * window[k] }
        }
        windowSquareSum = sum.map { max($0, 1e-8) }
    }

    /// The stem for a stereo mix (`[left, right]` at 44.1 kHz). Chunks overlap by half and run
    /// `chunksAtOnce` at a time; every sample gets exactly two contributions, so the result does
    /// not depend on the order chunks finish in.
    func stem(of mix: [[Float]], progress: (Double) -> Void) async throws -> [[Float]] {
        let chunk = Self.chunkSamples
        let step = chunk / 2
        let fade = chunk / 10
        let border = chunk - step
        let count = mix[0].count
        let padded = mix.map { reflectPad($0, border) }
        let total = padded[0].count
        let starts = Array(stride(from: 0, to: total, by: step))

        var fadeWindow = [Float](repeating: 1, count: chunk)
        for k in 0..<fade {
            let v = Float(k) / Float(fade - 1)
            fadeWindow[k] = v
            fadeWindow[chunk - 1 - k] = v
        }

        var sum = [[Float]](repeating: [Float](repeating: 0, count: total), count: 2)
        var weight = [Float](repeating: 0, count: total)
        var finished = 0
        func add(_ start: Int, _ stem: [[Float]]) {
            let length = min(chunk, total - start)
            var w = fadeWindow
            if start == 0 { for k in 0..<fade { w[k] = 1 } }
            if start + chunk >= total { for k in 0..<fade { w[chunk - 1 - k] = 1 } }
            for c in 0..<2 {
                for k in 0..<length { sum[c][start + k] += stem[c][k] * w[k] }
            }
            for k in 0..<length { weight[start + k] += w[k] }
            finished += 1
            progress(Double(finished) / Double(starts.count))
        }

        try await withThrowingTaskGroup(of: (Int, [[Float]]).self) { group in
            for (index, start) in starts.enumerated() {
                if index >= chunksAtOnce, let (done, stem) = try await group.next() {
                    add(done, stem)
                }
                group.addTask { (start, try await self.stem(ofChunk: Self.chunk(of: padded, at: start))) }
            }
            for try await (done, stem) in group {
                add(done, stem)
            }
        }
        return sum.map { channel in
            (0..<count).map { channel[border + $0] / max(weight[border + $0], 1e-8) }
        }
    }

    /// One `chunkSamples`-long slice, reflecting the song's tail out to full length.
    private static func chunk(of padded: [[Float]], at start: Int) -> [[Float]] {
        let length = min(chunkSamples, padded[0].count - start)
        return padded.map { channel in
            var x = Array(channel[start..<start + length])
            for k in length..<chunkSamples {
                let source = 2 * length - 2 - k
                x.append(source >= 0 ? x[source] : 0)
            }
            return x
        }
    }

    /// The stem for exactly one `chunkSamples`-long stereo chunk: the graph's native unit.
    func stem(ofChunk chunk: [[Float]]) async throws -> [[Float]] {
        let (n, h, f) = (Self.nFFT, hop, frames)
        var framed = [Float](repeating: 0, count: 2 * f * n)
        framed.withUnsafeMutableBufferPointer { dst in
            for c in 0..<2 {
                let x = reflectPad(chunk[c], Self.pad)
                x.withUnsafeBufferPointer { src in
                    for i in 0..<f {
                        (dst.baseAddress! + (c * f + i) * n).update(from: src.baseAddress! + i * h, count: n)
                    }
                }
            }
        }

        let outputs = try await model.run(["frames": MLTensor(shape: [1, 2, f, n], scalars: framed)])
        guard let recon = outputs["recon"] else { throw SplitError.model("the RoFormer model returned no recon output") }
        let values = await recon.values

        let length = n + h * (f - 1)
        return (0..<2).map { c in
            var acc = [Float](repeating: 0, count: length)
            for i in 0..<f {
                let base = i * h
                let source = (c * f + i) * n
                for k in 0..<n { acc[base + k] += values[source + k] }
            }
            for k in 0..<length { acc[k] /= windowSquareSum[k] }
            return Array(acc[Self.pad..<Self.pad + Self.chunkSamples])
        }
    }
}

/// numpy-style "reflect" padding (edge sample not repeated), mirroring repeatedly for short input.
func reflectPad(_ x: [Float], _ p: Int) -> [Float] {
    reflectPad(x, left: p, right: p)
}

func reflectPad(_ x: [Float], left: Int, right: Int) -> [Float] {
    let n = x.count
    guard n > 1 else { return [Float](repeating: x.first ?? 0, count: n + left + right) }
    let period = 2 * (n - 1)
    return (-left..<(n + right)).map { j in
        var m = abs(j) % period
        if m >= n { m = period - m }
        return x[m]
    }
}
