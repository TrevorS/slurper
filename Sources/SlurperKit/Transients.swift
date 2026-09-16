import Accelerate

/// Finds hits with spectral flux (Böck & Widmer's log-magnitude variant), placed to within a 64-sample block.
enum Transients {
    /// Audio kept before each attack when a hit is cut out.
    static let preroll = 0.003
    /// Onsets closer together than this are one hit.
    static let minimumGap = 0.05
    /// A hit is dropped when its first `minimumGap` of audio peaks below this fraction of the loudest sample (-40 dB).
    static let gate: Float = 0.01
    /// Level, relative to the loudest sample, below which audio counts as silence (-60 dB).
    static let silence: Float = 0.001
    /// Longest a cut-out hit runs.
    static let longestHit = 2.0

    static let frameSize = 1024
    static let hop = 256
    /// Log compression, `log(1 + compression * magnitude)`, with a full-scale sine at magnitude 1.
    private static let compression: Float = 100
    /// Normalized flux a peak must clear above its local mean.
    private static let delta: Float = 0.07
    /// Per bin, one over the width of a sixth-octave band there (at least one bin), so the few bins under a kick
    /// count as much as the hundreds above 2 kHz. DC counts for nothing.
    private static let binWeights: [Float] = (0..<frameSize / 2).map { $0 == 0 ? 0 : 1 / max(1, Float($0) * (pow(2, 1 / 6) - 1)) }

    /// Sample positions of attacks in mono audio, in order.
    static func onsets(of mono: [Float], sampleRate: Double) -> [Int] {
        let flux = spectralFlux(mono, bands: [0..<frameSize / 2])[0]
        guard let top = flux.max(), top > 0 else { return [] }
        let normalized = vDSP.divide(flux, top)
        let frames = { (seconds: Double) in max(1, Int((seconds * sampleRate / Double(hop)).rounded())) }
        let (preMax, postMax, preMean, postMean) = (frames(0.03), frames(0.01), frames(0.1), frames(0.07))
        var sums = [Double](repeating: 0, count: normalized.count + 1)
        for (t, value) in normalized.enumerated() { sums[t + 1] = sums[t] + Double(value) }

        let loudest = vDSP.maximumMagnitude(mono)
        let gap = Int(minimumGap * sampleRate)
        var attacks: [Int] = []
        for (t, value) in normalized.enumerated() where value > delta {
            guard value >= normalized[max(0, t - preMax)..<min(normalized.count, t + postMax + 1)].max()! else { continue }
            let (a, b) = (max(0, t - preMean), min(normalized.count, t + postMean + 1))
            guard Double(value) >= (sums[b] - sums[a]) / Double(b - a) + Double(delta) else { continue }
            let attack = refine(mono, center: t * hop)
            guard vDSP.maximumMagnitude(mono[attack..<min(mono.count, attack + gap)]) >= loudest * gate,
                  rises(mono, at: attack, sampleRate: sampleRate) else { continue }
            if let last = attacks.last, attack - last < gap { continue }
            attacks.append(attack)
        }
        return attacks
    }

    /// For each attack, from `preroll` before it to where the next hit's range starts (at most `longestHit`),
    /// less trailing 10 ms blocks of silence.
    static func hits(at attacks: [Int], in mono: [Float], sampleRate: Double) -> [Range<Int>] {
        guard !attacks.isEmpty else { return [] }
        let floor = vDSP.maximumMagnitude(mono) * silence
        let block = Int(0.01 * sampleRate)
        let starts = attacks.map { max(0, $0 - Int(preroll * sampleRate)) }
        return starts.indices.map { i in
            var end = min(i + 1 < starts.count ? starts[i + 1] : mono.count, attacks[i] + Int(longestHit * sampleRate))
            while end - block > attacks[i] + block, vDSP.maximumMagnitude(mono[end - block..<end]) < floor {
                end -= block
            }
            return starts[i]..<end
        }
    }

    static func mono(_ audio: [[Float]]) -> [Float] {
        guard let first = audio.first else { return [] }
        return vDSP.multiply(1 / Float(audio.count), audio.dropFirst().reduce(first) { vDSP.add($0, $1) })
    }

    /// For each band of FFT bins, the rise in log magnitude into frame `t` (centered on sample `t * hop`), summed
    /// with `binWeights`.
    static func spectralFlux(_ mono: [Float], bands: [Range<Int>]) -> [[Float]] {
        guard !mono.isEmpty else { return bands.map { _ in [] } }
        let bins = frameSize / 2
        let padding = [Float](repeating: 0, count: bins)
        let padded = padding + mono + padding
        let window = vDSP.window(ofType: Float.self, usingSequence: .hanningDenormalized, count: frameSize, isHalfWindow: false)
        let transform = try! vDSP.DiscreteFourierTransform(
            count: frameSize, direction: .forward, transformType: .complexComplex, ofType: Float.self)

        let zeros = [Float](repeating: 0, count: frameSize)
        var frame = zeros, real = zeros, imaginary = zeros
        var magnitude = padding, previous = padding, rise = padding
        let count = mono.count / hop + 1
        var flux = bands.map { _ in [Float](repeating: 0, count: count) }
        for t in 0..<count {
            vDSP.multiply(padded[t * hop..<t * hop + frameSize], window, result: &frame)
            transform.transform(inputReal: frame, inputImaginary: zeros, outputReal: &real, outputImaginary: &imaginary)
            vDSP.hypot(real[0..<bins], imaginary[0..<bins], result: &magnitude)
            vDSP.multiply(compression * 4 / Float(frameSize), magnitude, result: &magnitude)
            vForce.log1p(magnitude, result: &magnitude)
            vDSP.subtract(magnitude, previous, result: &rise)
            vDSP.clip(rise, to: 0...Float.greatestFiniteMagnitude, result: &rise)
            vDSP.multiply(rise, binWeights, result: &rise)
            for (b, band) in bands.enumerated() { flux[b][t] = vDSP.sum(rise[band]) }
            swap(&magnitude, &previous)
        }
        return flux
    }

    /// Whether the 10 ms after `attack` are louder than the 10 ms before. Sounds that stop abruptly leak
    /// broadband flux too, but get quieter.
    private static func rises(_ mono: [Float], at attack: Int, sampleRate: Double) -> Bool {
        let window = Int(0.01 * sampleRate)
        let before = mono[max(0, attack - window)..<attack]
        let after = mono[attack..<min(mono.count, attack + window)]
        return before.isEmpty || vDSP.rootMeanSquare(after) > 1.2 * vDSP.rootMeanSquare(before)
    }

    /// The attack behind a flux peak: within half a frame and a hop of the frame's center, the 64-sample block
    /// whose level jumps most over the four blocks before it, at the first sample reaching a tenth of that block's peak.
    private static func refine(_ mono: [Float], center: Int) -> Int {
        let block = 64
        let start = max(0, center - frameSize / 2 - hop)
        let end = min(mono.count, center + frameSize / 2 + hop)
        guard start < end else { return min(center, mono.count - 1) }
        let from = max(0, start - 4 * block)
        let levels = stride(from: from, to: end, by: block).map { vDSP.maximumMagnitude(mono[$0..<min(end, $0 + block)]) }

        var best = (start - from) / block
        var bestRatio: Float = 0
        for b in best..<levels.count {
            let ratio = levels[b] / max(levels[max(0, b - 4)..<b].max() ?? 0, 1e-4)
            if ratio > bestRatio { (best, bestRatio) = (b, ratio) }
        }
        let blockStart = from + best * block
        let threshold = levels[best] / 10
        return (blockStart..<min(end, blockStart + block)).first { abs(mono[$0]) >= threshold } ?? blockStart
    }
}
